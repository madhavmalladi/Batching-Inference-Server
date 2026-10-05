// Package integration runs the real gateway and fake worker over real gRPC
// sockets on localhost. Unlike the rest of the suite these tests use REAL time
// (sockets can't live in a synctest bubble), so they poll with timeouts
// instead of asserting exact instants, and the slow one is skipped by -short.
package integration

import (
	"context"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"batchserve/gateway/internal/batcher"
	"batchserve/gateway/internal/fakeworker"
	"batchserve/gateway/internal/metrics"
	"batchserve/gateway/internal/server"
	"batchserve/gateway/internal/testrig"
	"batchserve/gateway/internal/workerclient"
	pb "batchserve/gen/go/batchservev1"

	"github.com/prometheus/client_golang/prometheus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// startWorker serves a fake worker on addr ("127.0.0.1:0" picks a free port).
func startWorker(t *testing.T, addr string, cfg fakeworker.Config) (string, *fakeworker.Worker, func()) {
	t.Helper()
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	w := fakeworker.New(cfg)
	s := grpc.NewServer(grpc.MaxRecvMsgSize(256 << 20))
	pb.RegisterInferenceWorkerServer(s, w)
	go s.Serve(lis)
	stop := func() { s.Stop() }
	t.Cleanup(stop)
	return lis.Addr().String(), w, stop
}

type gateway struct {
	addr   string
	client pb.DetectorClient
	// stop mirrors cmd/gateway's shutdown order: stop the gRPC server first
	// (handlers still get answers because the batcher is alive), then the batcher.
	stop func()
}

func startGateway(t *testing.T, workerAddr string, cfg batcher.Config) *gateway {
	t.Helper()
	wc, err := workerclient.Dial(workerAddr)
	if err != nil {
		t.Fatal(err)
	}
	m := metrics.New(prometheus.NewRegistry())
	b, err := batcher.New(cfg, wc, m)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); b.Run(ctx) }()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	s := grpc.NewServer()
	server.Register(s, server.New(b, m, 2<<20))
	go s.Serve(lis)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	var once sync.Once
	g := &gateway{addr: lis.Addr().String(), client: pb.NewDetectorClient(conn)}
	g.stop = func() {
		once.Do(func() {
			s.GracefulStop()
			cancel()
			<-done
		})
	}
	// The test's own client connection and the worker client outlive the
	// gateway's shutdown, so probes after stop() see the SERVER refusing them,
	// not our own connection being closed (which reports CANCELED).
	t.Cleanup(func() { g.stop(); conn.Close(); wc.Close() })
	return g
}

func defaultCfg() batcher.Config {
	return batcher.Config{MaxBatchSize: 8, MaxWait: 5 * time.Millisecond, QueueSize: 256, MaxInflight: 1, WorkerTimeout: 5 * time.Second}
}

func detect(g *gateway, image []byte, timeout time.Duration) (*pb.DetectResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return g.client.Detect(ctx, &pb.DetectRequest{Image: image})
}

// eventually polls cond every few ms until it holds or the timeout passes.
func eventually(t *testing.T, timeout time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.After(timeout)
	tick := time.NewTicker(5 * time.Millisecond)
	defer tick.Stop()
	for !cond() {
		select {
		case <-deadline:
			t.Fatalf("timed out waiting for %s", what)
		case <-tick.C:
		}
	}
}

func TestIntegration_EndToEndWithFakeWorker(t *testing.T) {
	workerAddr, _, _ := startWorker(t, "127.0.0.1:0", fakeworker.Config{})
	g := startGateway(t, workerAddr, defaultCfg())

	const n = 50
	var wg sync.WaitGroup
	errs := make(chan string, n)
	for i := 1; i <= n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := detect(g, testrig.Image(i), 5*time.Second)
			switch {
			case err != nil:
				errs <- err.Error()
			case len(resp.Detections) != 1 || resp.Detections[0].Box.XMax != float32(i):
				errs <- "request got someone else's detection"
			}
		}()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Error(e)
	}
}

func TestIntegration_GatewayStartsBeforeWorker(t *testing.T) {
	// Reserve a free port, then release it so nothing is listening there yet.
	probe, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	workerAddr := probe.Addr().String()
	probe.Close()

	g := startGateway(t, workerAddr, defaultCfg())
	_, err = detect(g, testrig.Image(5), 2*time.Second)
	if got := status.Code(err); got != codes.Unavailable {
		t.Fatalf("with no worker: code = %v (%v), want UNAVAILABLE", got, err)
	}

	startWorker(t, workerAddr, fakeworker.Config{})
	// The gateway's gRPC client reconnects on its own (backoff capped at 1s);
	// no gateway restart is needed.
	eventually(t, 10*time.Second, "requests to succeed once the worker is up", func() bool {
		_, err := detect(g, testrig.Image(5), time.Second)
		return err == nil
	})
}

// throughput runs closed-loop clients against a fresh gateway for d and returns requests/second.
func throughput(t *testing.T, maxBatch int, d time.Duration) float64 {
	workerAddr, _, _ := startWorker(t, "127.0.0.1:0", fakeworker.Config{BaseLatency: 5 * time.Millisecond, PerImageLatency: 2 * time.Millisecond})
	cfg := defaultCfg()
	cfg.MaxBatchSize = maxBatch
	g := startGateway(t, workerAddr, cfg)

	var done atomic.Int64
	ctx, cancel := context.WithTimeout(context.Background(), d)
	defer cancel()
	var wg sync.WaitGroup
	for range 32 { // enough concurrency to fill a batch of 8
		wg.Add(1)
		go func() {
			defer wg.Done()
			for ctx.Err() == nil {
				if _, err := g.client.Detect(context.Background(), &pb.DetectRequest{Image: testrig.Image(100)}); err == nil && ctx.Err() == nil {
					done.Add(1)
				}
			}
		}()
	}
	wg.Wait()
	return float64(done.Load()) / d.Seconds()
}

func TestIntegration_BatchingBeatsNoBatching(t *testing.T) {
	if testing.Short() {
		t.Skip("uses real time (about 2s)")
	}
	// With the fake's 5ms + 2ms/image cost and one batch in flight, B=1 tops
	// out near 143 req/s and B=8 near 380 req/s: a ratio near 2.6 against a
	// threshold of 1.5, so scheduling noise won't flip it.
	b1 := throughput(t, 1, time.Second)
	b8 := throughput(t, 8, time.Second)
	t.Logf("B=1: %.0f req/s, B=8: %.0f req/s (%.2fx)", b1, b8, b8/b1)
	if b8 <= 1.5*b1 {
		t.Fatalf("B=8 throughput %.0f is not > 1.5x B=1 throughput %.0f", b8, b1)
	}
}

func TestIntegration_GracefulShutdown(t *testing.T) {
	workerAddr, w, _ := startWorker(t, "127.0.0.1:0", fakeworker.Config{BaseLatency: 300 * time.Millisecond})
	cfg := defaultCfg()
	cfg.MaxBatchSize = 10
	g := startGateway(t, workerAddr, cfg)

	const n = 10
	results := make(chan error, n)
	for i := 1; i <= n; i++ {
		go func() { _, err := detect(g, testrig.Image(i), 5*time.Second); results <- err }()
	}
	eventually(t, 3*time.Second, "the batch to reach the worker", func() bool { return len(w.Calls()) > 0 })

	// SIGTERM-equivalent while the batch is at the (slow) worker.
	stopped := make(chan struct{})
	go func() { g.stop(); close(stopped) }()

	// New requests must now be refused. Right at the start of the shutdown a
	// request can still slip in, so poll until the refusal shows up.
	eventually(t, 2*time.Second, "new requests to get UNAVAILABLE", func() bool {
		_, err := detect(g, testrig.Image(5), time.Second)
		return status.Code(err) == codes.Unavailable
	})

	for range n {
		if err := <-results; err != nil {
			t.Errorf("in-flight request failed during graceful shutdown: %v", err)
		}
	}
	<-stopped
}
