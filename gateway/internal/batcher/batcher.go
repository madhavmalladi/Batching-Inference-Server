// Package batcher is the heart of the gateway: it groups concurrent requests
// into batches and sends each batch to the worker once.
//
// The concurrency model, in one picture:
//
//	N handler goroutines ──try-send──▶ [ queue: chan, cap Q ] ──▶ 1 batcher goroutine (Run)
//	      ▲                                                          │ 1. acquire a slot (sem, cap F)
//	      │ per-request result chan (cap 1)                          ▼ 2. form a batch
//	      └───────────────────────────────────────────────── 1 dispatch goroutine per batch
//
// Who owns what (this is why there are almost no locks):
//   - The queue channel is the only thing handlers and the batcher share. A Go
//     channel is already a thread-safe bounded FIFO.
//   - The batch being formed lives in local variables of ONE goroutine (Run), so
//     it needs no lock.
//   - Each request owns its own reply channel, so routing a result back needs
//     no shared map.
package batcher

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"batchserve/gateway/internal/metrics"
	pb "batchserve/gen/go/batchservev1"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Inferer is the batcher's view of the worker. It has two real implementations:
// the gRPC client (internal/workerclient) and the in-process fake
// (internal/fakeworker). Its signature matches the generated server method, so
// the fake satisfies it with no adapter.
type Inferer interface {
	InferBatch(ctx context.Context, req *pb.InferBatchRequest) (*pb.InferBatchResponse, error)
}

// Config holds the batching parameters from CLAUDE.md.
type Config struct {
	MaxBatchSize  int           // B: dispatch when a batch reaches this size (1 = no batching)
	MaxWait       time.Duration // W: ...or when the oldest request has waited this long (0 = greedy)
	QueueSize     int           // Q: admission queue capacity
	MaxInflight   int           // F: batches allowed to be at the worker at once
	WorkerTimeout time.Duration // fixed deadline for every InferBatch call
}

func (c Config) Validate() error {
	switch {
	case c.MaxBatchSize < 1 || c.MaxBatchSize > 64:
		return fmt.Errorf("max batch size %d out of range 1..64", c.MaxBatchSize)
	case c.MaxWait < 0:
		return fmt.Errorf("max wait %v is negative", c.MaxWait)
	case c.QueueSize < 1:
		return fmt.Errorf("queue size %d must be >= 1", c.QueueSize)
	case c.MaxInflight < 1:
		return fmt.Errorf("max inflight batches %d must be >= 1", c.MaxInflight)
	case c.WorkerTimeout <= 0:
		return fmt.Errorf("worker timeout %v must be positive", c.WorkerTimeout)
	}
	return nil
}

// Result is what a successful request gets back.
type Result struct {
	Detections []*pb.Detection
	Timing     *pb.ServerTiming
}

// outcome is a Result or an error; it is what travels on a request's reply channel.
type outcome struct {
	Result
	Err error
}

// pending is one admitted request. A batch is a []*pending.
type pending struct {
	id         uint64
	image      []byte
	enqueuedAt time.Time

	// result has capacity 1 so the dispatcher's send NEVER blocks, even if the
	// client has already given up and nobody will ever receive. With an
	// unbuffered channel that send would block forever: a leaked goroutine that
	// also holds the batch's in-flight slot, eventually stalling the server.
	result chan outcome
}

type Batcher struct {
	cfg     Config
	inferer Inferer
	m       *metrics.Metrics

	queue  chan *pending
	sem    chan struct{} // counting semaphore: a send acquires a slot, a receive releases it
	nextID atomic.Uint64 // request IDs; atomic because every handler goroutine calls Submit
	wg     sync.WaitGroup
}

func New(cfg Config, inferer Inferer, m *metrics.Metrics) (*Batcher, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	b := &Batcher{
		cfg:     cfg,
		inferer: inferer,
		m:       m,
		queue:   make(chan *pending, cfg.QueueSize),
		sem:     make(chan struct{}, cfg.MaxInflight),
	}
	m.BindQueueDepth(func() float64 { return float64(len(b.queue)) })
	return b, nil
}

// Submit admits one request and waits for its result. It runs in the caller's
// own goroutine (one per gRPC call), so blocking here is cheap and simple.
func (b *Batcher) Submit(ctx context.Context, image []byte) (Result, error) {
	p := &pending{
		id:         b.nextID.Add(1),
		image:      image,
		enqueuedAt: time.Now(),
		result:     make(chan outcome, 1),
	}

	// Admission rule: a NON-blocking send. If the queue is full we say so
	// immediately instead of parking the caller. Blocking here would turn a
	// full queue into unbounded waiting, which is exactly what backpressure
	// exists to prevent (docs/design.md §6).
	select {
	case b.queue <- p:
	default:
		b.m.Rejected.WithLabelValues(metrics.RejectQueueFull).Inc()
		return Result{}, status.Error(codes.ResourceExhausted, "queue full")
	}

	o := <-p.result
	return o.Result, o.Err
}

// Run is the single batcher goroutine. It returns when ctx is cancelled and
// every in-flight batch has finished.
func (b *Batcher) Run(ctx context.Context) {
	defer b.wg.Wait() // don't return while dispatch goroutines are still using the worker

	var batchID uint64
	for {
		// RULE 1: take an in-flight slot BEFORE forming a batch. If we formed
		// the batch first and then waited for a slot, requests arriving during
		// that wait could not join it. Waiting here lets the queue build up, so
		// the next batch comes out fuller.
		select {
		case b.sem <- struct{}{}:
		case <-ctx.Done():
			return
		}

		batch, reason, ok := b.formBatch(ctx)
		if !ok { // shutting down
			<-b.sem
			return
		}

		batchID++
		b.wg.Add(1)
		go b.dispatch(batchID, batch, reason) // releases the slot when done
	}
}

// formBatch blocks for the first request, then fills the batch until it is
// full or the oldest request's deadline arrives. This function is the policy.
func (b *Batcher) formBatch(ctx context.Context) (batch []*pending, reason pb.DispatchReason, ok bool) {
	var first *pending
	select {
	case first = <-b.queue:
	case <-ctx.Done():
		return nil, 0, false
	}
	batch = append(make([]*pending, 0, b.cfg.MaxBatchSize), first)

	// RULE 2: the batch deadline counts from when the OLDEST request was
	// admitted, not from now. Under a backlog this is already in the past, so
	// we send immediately instead of making queued requests wait another W.
	deadline := first.enqueuedAt.Add(b.cfg.MaxWait)

	// Step 1: take what is already queued, WITHOUT blocking. This must come
	// before the timer: if both the deadline and a queued request were ready
	// in one select, Go would pick randomly, and a backlogged batch would
	// sometimes leave requests behind for no reason.
drain:
	for len(batch) < b.cfg.MaxBatchSize {
		select {
		case p := <-b.queue:
			batch = append(batch, p)
		default:
			break drain
		}
	}
	if len(batch) == b.cfg.MaxBatchSize {
		return batch, pb.DispatchReason_DISPATCH_REASON_FULL, true
	}

	// Step 2: wait for more requests or the deadline, whichever is first. A
	// timer created with a past deadline fires immediately (W=0 and backlog).
	timer := time.NewTimer(time.Until(deadline))
	defer timer.Stop()
	for len(batch) < b.cfg.MaxBatchSize {
		select {
		case p := <-b.queue:
			batch = append(batch, p)
		case <-timer.C:
			return batch, pb.DispatchReason_DISPATCH_REASON_TIMEOUT, true
		case <-ctx.Done():
			return batch, 0, false
		}
	}
	return batch, pb.DispatchReason_DISPATCH_REASON_FULL, true
}

// dispatch runs in its own goroutine so the batcher can immediately start
// forming the next batch (up to F batches overlap at the worker).
func (b *Batcher) dispatch(id uint64, batch []*pending, reason pb.DispatchReason) {
	defer b.wg.Done()
	// Deferred first = runs last: the slot is released only after every result
	// has been written, so F really bounds the work at the worker.
	defer func() { <-b.sem }()

	sent := time.Now()
	req := &pb.InferBatchRequest{BatchId: id, Images: make([]*pb.WorkerImage, len(batch))}
	for i, p := range batch {
		req.Images[i] = &pb.WorkerImage{RequestId: p.id, Data: p.image}
		b.m.QueueWait.Observe(sent.Sub(p.enqueuedAt).Seconds())
	}
	b.m.BatchSize.Observe(float64(len(batch)))
	b.m.BatchesTotal.WithLabelValues(reasonLabel(reason)).Inc()

	// A FIXED timeout, deliberately not the batch's earliest client deadline:
	// one impatient client must not kill a batch shared with patient ones. The
	// parent is Background (not Run's ctx) so a graceful shutdown lets
	// in-flight batches finish instead of aborting them.
	ctx, cancel := context.WithTimeout(context.Background(), b.cfg.WorkerTimeout)
	defer cancel()
	resp, err := b.inferer.InferBatch(ctx, req)
	took := time.Since(sent)

	b.m.BatchDuration.Observe(took.Seconds())
	if resp != nil {
		b.m.WorkerInference.Observe(float64(resp.GetTiming().GetInferenceUs()) / 1e6)
	}

	outcomes := route(batch, resp, err)
	for i, p := range batch {
		if outcomes[i].Err == nil {
			outcomes[i].Timing = &pb.ServerTiming{
				BatchSize:      uint32(len(batch)),
				QueueWaitUs:    sent.Sub(p.enqueuedAt).Microseconds(),
				BatchLatencyUs: took.Microseconds(),
				DispatchReason: reason,
			}
		}
		p.result <- outcomes[i] // never blocks: capacity 1, and this is the only send
	}
}

// route turns the worker's answer into one outcome per request, by position.
// The response is validated as a whole BEFORE any result is used: if the count
// or any ID is wrong, we cannot trust any position, so every request fails.
// Failing loudly beats silently handing client A client B's detections.
func route(batch []*pending, resp *pb.InferBatchResponse, err error) []outcome {
	out := make([]outcome, len(batch))
	failAll := func(e error) []outcome {
		for i := range out {
			out[i].Err = e
		}
		return out
	}

	if err != nil {
		return failAll(status.Errorf(codes.Unavailable, "worker call failed: %v", err))
	}
	results := resp.GetResults()
	if len(results) != len(batch) {
		return failAll(status.Errorf(codes.Internal, "worker returned %d results for %d images", len(results), len(batch)))
	}
	for i, p := range batch {
		if results[i].GetRequestId() != p.id {
			return failAll(status.Errorf(codes.Internal, "worker result %d has request_id %d, want %d", i, results[i].GetRequestId(), p.id))
		}
	}

	for i, r := range results {
		if r.Error != "" {
			// Only this client's image was bad; its neighbours are fine.
			out[i].Err = status.Error(codes.InvalidArgument, r.Error)
			continue
		}
		out[i].Detections = r.Detections
	}
	return out
}

func reasonLabel(r pb.DispatchReason) string {
	if r == pb.DispatchReason_DISPATCH_REASON_FULL {
		return metrics.ReasonFull
	}
	return metrics.ReasonTimeout
}
