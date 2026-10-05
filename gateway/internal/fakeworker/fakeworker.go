// Package fakeworker is a stand-in for the C++ inference worker: it sleeps
// instead of running a model, and returns detections derived from the image
// bytes so tests can prove every client got *its own* answer.
//
// The type implements pb.InferenceWorkerServer, so cmd/fakeworker only has to
// register it with a gRPC server. Gateway tests also call InferBatch directly,
// in-process, behind the batcher's Inferer interface (no network at all).
// See docs/fake-worker.md for the spec this follows.
package fakeworker

import (
	"bytes"
	"context"
	"fmt"
	"hash/crc32"
	"log/slog"
	"math/rand"
	"sync"
	"time"

	pb "batchserve/gen/go/batchservev1"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Config mirrors the cmd/fakeworker flags.
type Config struct {
	BaseLatency     time.Duration // fixed cost per batch
	PerImageLatency time.Duration // added cost per image
	Jitter          float64       // 0.1 = sleep is scaled by U(0.9, 1.1)
	FailRate        float64       // probability a whole batch returns UNAVAILABLE
	Seed            int64         // seeds the RNG so runs are reproducible
	Logger          *slog.Logger  // nil = discard
}

// BatchRecord is what the worker saw for one InferBatch call (test observability).
type BatchRecord struct {
	Size int
	IDs  []uint64
	At   time.Time
}

type Worker struct {
	pb.UnimplementedInferenceWorkerServer

	// Block, if non-nil, is waited on at the start of every InferBatch (before
	// the sleep). Tests use it to hold the worker "busy": close(Block) releases
	// every waiting and future call. Set it before the first call; it is read
	// without a lock.
	Block chan struct{}

	cfg Config
	log *slog.Logger

	// One mutex guards everything below. The RNG is not goroutine-safe, and
	// concurrent InferBatch calls (max_inflight > 1) are exactly the case we
	// need to support, so every draw happens under mu. The critical sections
	// are tiny; this is never a bottleneck.
	mu      sync.Mutex
	rng     *rand.Rand
	calls   []BatchRecord
	current int // InferBatch calls running right now
	maxConc int // high-water mark of current
}

func New(cfg Config) *Worker {
	log := cfg.Logger
	if log == nil {
		log = slog.New(slog.DiscardHandler)
	}
	return &Worker{cfg: cfg, log: log, rng: rand.New(rand.NewSource(cfg.Seed))}
}

// Calls returns a copy of every batch received so far, in arrival order.
func (w *Worker) Calls() []BatchRecord {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]BatchRecord(nil), w.calls...)
}

// MaxConcurrent is the most InferBatch calls that were ever running at once.
// A call that is parked on Block counts as running.
func (w *Worker) MaxConcurrent() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.maxConc
}

// begin records the call and the concurrency high-water mark, and draws the
// random numbers for this batch. Drawing both numbers in one place, in a fixed
// order, keeps a given seed reproducible.
func (w *Worker) begin(req *pb.InferBatchRequest) (fail bool, scale float64) {
	w.mu.Lock()
	defer w.mu.Unlock()
	rec := BatchRecord{Size: len(req.Images), At: time.Now()}
	for _, img := range req.Images {
		rec.IDs = append(rec.IDs, img.RequestId)
	}
	w.calls = append(w.calls, rec)
	w.current++
	w.maxConc = max(w.maxConc, w.current)

	fail = w.rng.Float64() < w.cfg.FailRate // Float64 is in [0,1): rate 0 never fails, rate 1 always does
	scale = 1 + (w.rng.Float64()*2-1)*w.cfg.Jitter
	return fail, scale
}

func (w *Worker) end() {
	w.mu.Lock()
	w.current--
	w.mu.Unlock()
}

// InferBatch implements the worker contract in proto/batchserve/v1/worker.proto.
func (w *Worker) InferBatch(ctx context.Context, req *pb.InferBatchRequest) (*pb.InferBatchResponse, error) {
	fail, scale := w.begin(req)
	defer w.end()

	if w.Block != nil {
		select {
		case <-w.Block:
		case <-ctx.Done():
			return nil, status.FromContextError(ctx.Err()).Err()
		}
	}
	if fail {
		return nil, status.Error(codes.Unavailable, "fakeworker: injected failure")
	}

	n := len(req.Images)
	resp := &pb.InferBatchResponse{BatchId: req.BatchId, Timing: &pb.WorkerTiming{}}
	if n == 0 {
		return resp, nil
	}

	// sleep = (base + per_image × N) × (1 + U(-jitter, +jitter))
	sleep := time.Duration(float64(w.cfg.BaseLatency+time.Duration(n)*w.cfg.PerImageLatency) * scale)
	if sleep > 0 {
		// A timer + select (not time.Sleep) so a cancelled ctx ends the sleep at once.
		timer := time.NewTimer(sleep)
		defer timer.Stop()
		select {
		case <-timer.C:
		case <-ctx.Done():
			return nil, status.FromContextError(ctx.Err()).Err()
		}
	}
	resp.Timing.InferenceUs = sleep.Microseconds()

	for _, img := range req.Images {
		resp.Results = append(resp.Results, detect(img))
	}
	w.log.Info("batch", "batch_id", req.BatchId, "size", n, "slept_ms", float64(sleep.Microseconds())/1000)
	return resp, nil
}

var badPrefix = []byte("BADIMG")

// detect fakes the model: the answer depends only on the image bytes.
func detect(img *pb.WorkerImage) *pb.ImageResult {
	res := &pb.ImageResult{RequestId: img.RequestId}
	if len(img.Data) == 0 || bytes.HasPrefix(img.Data, badPrefix) {
		res.Error = "fake decode error"
		return res
	}
	classID := int32(crc32.ChecksumIEEE(img.Data) % 80)
	size := float32(len(img.Data))
	res.Detections = []*pb.Detection{{
		Box:       &pb.BoundingBox{XMin: 0, YMin: 0, XMax: size, YMax: size},
		ClassId:   classID,
		ClassName: fmt.Sprintf("class_%d", classID),
		Score:     0.5,
	}}
	return res
}
