// Package server is the public Detector gRPC service. It owns request
// validation and metrics about requests; the batcher owns queueing and dispatch.
package server

import (
	"context"
	"time"

	"batchserve/gateway/internal/batcher"
	"batchserve/gateway/internal/metrics"
	pb "batchserve/gen/go/batchservev1"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"
)

type Server struct {
	pb.UnimplementedDetectorServer
	batcher       *batcher.Batcher
	m             *metrics.Metrics
	maxImageBytes int
}

func New(b *batcher.Batcher, m *metrics.Metrics, maxImageBytes int) *Server {
	return &Server{batcher: b, m: m, maxImageBytes: maxImageBytes}
}

// Register wires the Detector service onto s, plus the standard health service
// (for compose healthchecks) and gRPC reflection (so `grpcurl` works without
// the .proto). The returned health server lets main flip to NOT_SERVING on shutdown.
func Register(s *grpc.Server, d *Server) *health.Server {
	pb.RegisterDetectorServer(s, d)
	h := health.NewServer() // reports SERVING for the "" (whole server) service by default
	healthpb.RegisterHealthServer(s, h)
	reflection.Register(s)
	return h
}

// Detect handles one image. grpc-go calls it in a fresh goroutine per RPC.
func (s *Server) Detect(ctx context.Context, req *pb.DetectRequest) (*pb.DetectResponse, error) {
	start := time.Now()
	resp, err := s.detect(ctx, req)
	// One place records duration + code for every outcome, including rejections,
	// so the per-code series always add up to the total request count.
	s.m.RequestDuration.WithLabelValues(metrics.CodeName(status.Code(err))).Observe(time.Since(start).Seconds())
	return resp, err
}

func (s *Server) detect(ctx context.Context, req *pb.DetectRequest) (*pb.DetectResponse, error) {
	// Admission rule: validate before queueing, so bad input never takes a queue slot.
	if n := len(req.GetImage()); n == 0 || n > s.maxImageBytes {
		s.m.Rejected.WithLabelValues(metrics.RejectInvalid).Inc()
		if n == 0 {
			return nil, status.Error(codes.InvalidArgument, "image is empty")
		}
		return nil, status.Errorf(codes.InvalidArgument, "image is %d bytes, limit is %d", n, s.maxImageBytes)
	}

	res, err := s.batcher.Submit(ctx, req.Image)
	if err != nil {
		return nil, err
	}
	return &pb.DetectResponse{Detections: res.Detections, Timing: res.Timing}, nil
}
