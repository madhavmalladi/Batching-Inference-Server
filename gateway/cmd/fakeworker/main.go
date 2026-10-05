// Command fakeworker is a stand-in for the C++ worker that sleeps instead of
// running a model. It is a thin gRPC wrapper around internal/fakeworker; see
// docs/fake-worker.md for the flags and behaviour.
package main

import (
	"flag"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"batchserve/gateway/internal/fakeworker"
	pb "batchserve/gen/go/batchservev1"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

func main() {
	var (
		listen   = flag.String("listen", ":50052", "address to listen on")
		base     = flag.Duration("base-latency", 5*time.Millisecond, "fixed cost per batch")
		perImage = flag.Duration("per-image-latency", 2*time.Millisecond, "added cost per image")
		jitter   = flag.Float64("jitter", 0, "fraction: sleep is scaled by a uniform random factor in [1-jitter, 1+jitter]")
		failRate = flag.Float64("fail-rate", 0, "probability that a whole batch fails with UNAVAILABLE")
		seed     = flag.Int64("seed", 1, "RNG seed for jitter and fail-rate")
	)
	flag.Parse()
	log := slog.New(slog.NewTextHandler(os.Stderr, nil))

	if *jitter < 0 || *jitter >= 1 || *failRate < 0 || *failRate > 1 {
		log.Error("invalid flags: want 0 <= jitter < 1 and 0 <= fail-rate <= 1")
		os.Exit(2)
	}

	lis, err := net.Listen("tcp", *listen)
	if err != nil {
		log.Error("listen failed", "err", err)
		os.Exit(1)
	}

	w := fakeworker.New(fakeworker.Config{
		BaseLatency: *base, PerImageLatency: *perImage,
		Jitter: *jitter, FailRate: *failRate, Seed: *seed, Logger: log,
	})
	// Same 256 MiB limit as the real worker: gRPC's 4 MiB default is too small
	// for a batch of images, and would surface as a confusing RESOURCE_EXHAUSTED.
	s := grpc.NewServer(grpc.MaxRecvMsgSize(256<<20), grpc.MaxSendMsgSize(256<<20))
	pb.RegisterInferenceWorkerServer(s, w)
	healthpb.RegisterHealthServer(s, health.NewServer()) // SERVING immediately
	reflection.Register(s)

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sig
		log.Info("shutting down")
		s.GracefulStop()
	}()

	log.Info("fakeworker listening", "addr", lis.Addr().String(),
		"base_latency", *base, "per_image_latency", *perImage, "jitter", *jitter, "fail_rate", *failRate, "seed", *seed)
	if err := s.Serve(lis); err != nil {
		log.Error("serve failed", "err", err)
		os.Exit(1)
	}
}
