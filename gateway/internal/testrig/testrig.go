// Package testrig wires a fake worker, batcher, metrics and Detector handler
// together in-process, for tests only. It exists so the batcher, server and
// metrics tests share one setup instead of three copies.
//
// Everything here is meant to run inside a testing/synctest bubble: it starts
// goroutines, and Close stops them, which is what lets the bubble end cleanly
// (a goroutine that never exits would make synctest.Test report a deadlock).
package testrig

import (
	"context"
	"time"

	"batchserve/gateway/internal/batcher"
	"batchserve/gateway/internal/fakeworker"
	"batchserve/gateway/internal/metrics"
	"batchserve/gateway/internal/server"
	pb "batchserve/gen/go/batchservev1"

	"github.com/prometheus/client_golang/prometheus"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type Options struct {
	Batcher       batcher.Config // zero QueueSize/MaxInflight/WorkerTimeout get defaults
	Worker        fakeworker.Config
	Inferer       batcher.Inferer // if set, replaces the fake worker (for stub workers)
	MaxImageBytes int             // default 2 MiB
}

type Rig struct {
	Worker   *fakeworker.Worker // always present, even if Options.Inferer overrides it
	Batcher  *batcher.Batcher
	Server   *server.Server
	Metrics  *metrics.Metrics
	Registry *prometheus.Registry
	Start    time.Time // virtual time at creation; subtract it to get "t=10ms"

	cancel context.CancelFunc
	done   chan struct{}
}

func New(opts Options) *Rig {
	cfg := opts.Batcher
	if cfg.QueueSize == 0 {
		cfg.QueueSize = 256
	}
	if cfg.MaxInflight == 0 {
		cfg.MaxInflight = 1
	}
	if cfg.WorkerTimeout == 0 {
		cfg.WorkerTimeout = 10 * time.Second
	}
	if opts.MaxImageBytes == 0 {
		opts.MaxImageBytes = 2 << 20
	}

	r := &Rig{Worker: fakeworker.New(opts.Worker), Registry: prometheus.NewRegistry(), Start: time.Now(), done: make(chan struct{})}
	r.Metrics = metrics.New(r.Registry)
	var inf batcher.Inferer = r.Worker
	if opts.Inferer != nil {
		inf = opts.Inferer
	}
	b, err := batcher.New(cfg, inf, r.Metrics)
	if err != nil {
		panic(err) // a bad config in a test is a bug in the test
	}
	r.Batcher = b
	r.Server = server.New(b, r.Metrics, opts.MaxImageBytes)

	ctx, cancel := context.WithCancel(context.Background())
	r.cancel = cancel
	go func() {
		defer close(r.done)
		b.Run(ctx)
	}()
	return r
}

// Stop asks the batcher to shut down without waiting for it (in-flight batches
// keep running until they finish).
func (r *Rig) Stop() { r.cancel() }

// Close stops the batcher and waits for it, so no goroutine outlives the test.
func (r *Rig) Close() {
	r.cancel()
	<-r.done
}

// Elapsed is the virtual time since the rig was created.
func (r *Rig) Elapsed() time.Duration { return time.Since(r.Start) }

// Call is one in-flight Detect request.
type Call struct {
	Done chan struct{} // closed when the request finished
	Resp *pb.DetectResponse
	Err  error
	At   time.Duration // virtual time (since rig start) when it finished
}

// Code is the gRPC status code of the call (OK if it succeeded).
func (c *Call) Code() codes.Code { return status.Code(c.Err) }

// Go starts a Detect call in its own goroutine, like grpc-go does for a real RPC.
func (r *Rig) Go(ctx context.Context, image []byte) *Call {
	c := &Call{Done: make(chan struct{})}
	go func() {
		defer close(c.Done)
		c.Resp, c.Err = r.Server.Detect(ctx, &pb.DetectRequest{Image: image})
		c.At = r.Elapsed()
	}()
	return c
}

// Image returns n bytes that the fake worker treats as a valid image. Its
// length is echoed back as box.x_max, so tests can check routing.
func Image(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = 'x'
	}
	return b
}
