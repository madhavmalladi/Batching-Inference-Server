// Command gateway is the public batching front end. It wires the pieces
// together and owns process lifecycle; the interesting logic is in internal/batcher.
package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"batchserve/gateway/internal/batcher"
	"batchserve/gateway/internal/metrics"
	"batchserve/gateway/internal/server"
	"batchserve/gateway/internal/workerclient"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"google.golang.org/grpc"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
)

func main() {
	var (
		listen        = flag.String("listen", ":50051", "gRPC listen address (Detector service)")
		metricsListen = flag.String("metrics-listen", ":2112", "HTTP listen address for /metrics")
		workerAddr    = flag.String("worker-addr", "localhost:50052", "inference worker address")
		cfg           batcher.Config
		maxImageBytes = flag.Int("max-image-bytes", 2097152, "largest accepted image, in bytes")
	)
	flag.IntVar(&cfg.MaxBatchSize, "max-batch-size", 8, "B: dispatch when a batch reaches this size (1..64; 1 = no batching)")
	flag.DurationVar(&cfg.MaxWait, "max-wait", 5*time.Millisecond, "W: dispatch when the oldest request has waited this long (0 = greedy)")
	flag.IntVar(&cfg.QueueSize, "queue-size", 256, "Q: admission queue capacity; beyond it requests get RESOURCE_EXHAUSTED")
	flag.IntVar(&cfg.MaxInflight, "max-inflight-batches", 1, "F: batches allowed at the worker at once")
	flag.DurationVar(&cfg.WorkerTimeout, "worker-timeout", 10*time.Second, "fixed deadline for each InferBatch call")
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stderr, nil))
	if err := cfg.Validate(); err != nil {
		log.Error("invalid flags", "err", err)
		os.Exit(2)
	}
	if *maxImageBytes < 1 {
		log.Error("invalid flags", "err", "max-image-bytes must be >= 1")
		os.Exit(2)
	}

	if err := run(log, cfg, *listen, *metricsListen, *workerAddr, *maxImageBytes); err != nil {
		log.Error("gateway failed", "err", err)
		os.Exit(1)
	}
}

func run(log *slog.Logger, cfg batcher.Config, listen, metricsListen, workerAddr string, maxImageBytes int) error {
	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))
	m := metrics.New(reg)

	wc, err := workerclient.Dial(workerAddr)
	if err != nil {
		return err
	}
	defer wc.Close()

	b, err := batcher.New(cfg, wc, m)
	if err != nil {
		return err
	}

	// The batcher gets its OWN context rather than the signal context. On
	// SIGTERM we must stop accepting RPCs and let the handlers that are waiting
	// on the batcher finish; if the batcher died with the signal, they'd all
	// fail. So the order is: stop the gRPC server first, THEN stop the batcher.
	batcherCtx, stopBatcher := context.WithCancel(context.Background())
	batcherDone := make(chan struct{})
	go func() {
		defer close(batcherDone)
		b.Run(batcherCtx)
	}()

	lis, err := net.Listen("tcp", listen)
	if err != nil {
		stopBatcher()
		return err
	}
	// The default 4 MiB receive limit would reject images the flag allows.
	grpcServer := grpc.NewServer(grpc.MaxRecvMsgSize(maxImageBytes + 1<<20))
	health := server.Register(grpcServer, server.New(b, m, maxImageBytes))

	metricsSrv := &http.Server{Addr: metricsListen, Handler: metricsMux(reg)}
	go func() {
		if err := metricsSrv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
			log.Error("metrics server failed", "err", err)
		}
	}()

	serveErr := make(chan error, 1)
	go func() { serveErr <- grpcServer.Serve(lis) }()
	log.Info("gateway listening", "grpc", lis.Addr().String(), "metrics", metricsListen, "worker", workerAddr,
		"max_batch_size", cfg.MaxBatchSize, "max_wait", cfg.MaxWait, "queue_size", cfg.QueueSize,
		"max_inflight_batches", cfg.MaxInflight, "worker_timeout", cfg.WorkerTimeout)

	sig, stopSignals := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stopSignals()
	select {
	case <-sig.Done():
		log.Info("shutting down")
	case err := <-serveErr:
		stopBatcher()
		<-batcherDone
		return err
	}

	// 1. Tell health checkers we're leaving, then stop accepting new RPCs and
	//    wait for in-flight ones (the batcher is still running, so they finish).
	health.SetServingStatus("", healthpb.HealthCheckResponse_NOT_SERVING)
	stopped := make(chan struct{})
	go func() { grpcServer.GracefulStop(); close(stopped) }()
	select {
	case <-stopped:
	case <-time.After(cfg.WorkerTimeout + 5*time.Second): // a hung worker must not hang shutdown
		log.Warn("graceful stop timed out; forcing")
		grpcServer.Stop()
	}
	// 2. Now nothing is waiting on the batcher; stop it.
	stopBatcher()
	<-batcherDone

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	return metricsSrv.Shutdown(ctx)
}

func metricsMux(g prometheus.Gatherer) http.Handler {
	mux := http.NewServeMux()
	mux.Handle("/metrics", metrics.Handler(g))
	return mux
}
