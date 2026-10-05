package server_test

import (
	"context"
	"slices"
	"testing"
	"testing/synctest"
	"time"

	"batchserve/gateway/internal/batcher"
	"batchserve/gateway/internal/metrics"
	"batchserve/gateway/internal/testrig"
	pb "batchserve/gen/go/batchservev1"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// overloadedRig builds the backpressure scenario from docs/test-plan.md: Q=4,
// B=1, F=1, worker blocked. It returns the 5 requests the system can hold: one
// in flight at the worker plus four queued (Q + B×F).
func overloadedRig() (*testrig.Rig, []*testrig.Call) {
	r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 1, QueueSize: 4, MaxInflight: 1}})
	r.Worker.Block = make(chan struct{})
	var held []*testrig.Call
	for i := range 5 {
		held = append(held, r.Go(context.Background(), testrig.Image(10+i)))
		// Let each request settle before sending the next, so request 1 is
		// definitely at the worker (not still queued) when 2..5 arrive.
		synctest.Wait()
	}
	return r, held
}

func TestDetect_QueueOverflowRejectsWithResourceExhausted(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r, held := overloadedRig()
		defer r.Close()

		_, err := r.Server.Detect(context.Background(), &pb.DetectRequest{Image: testrig.Image(99)})
		if got := status.Code(err); got != codes.ResourceExhausted {
			t.Fatalf("6th request code = %v, want RESOURCE_EXHAUSTED", got)
		}
		if got := testutil.ToFloat64(r.Metrics.Rejected.WithLabelValues(metrics.RejectQueueFull)); got != 1 {
			t.Errorf("rejected{queue_full} = %v, want 1", got)
		}

		close(r.Worker.Block)
		waitAll(held...)
	})
}

func TestDetect_RejectionIsImmediate(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r, held := overloadedRig()
		defer r.Close()

		// Called synchronously, with no synctest.Wait: if admission ever
		// blocked, this call would never return and the bubble would deadlock.
		start := time.Now()
		_, err := r.Server.Detect(context.Background(), &pb.DetectRequest{Image: testrig.Image(99)})
		if status.Code(err) != codes.ResourceExhausted {
			t.Fatalf("code = %v, want RESOURCE_EXHAUSTED", status.Code(err))
		}
		if took := time.Since(start); took != 0 {
			t.Errorf("rejection took %v of virtual time, want 0", took)
		}

		close(r.Worker.Block)
		waitAll(held...)
	})
}

func TestDetect_RejectedRequestNeverReachesWorker(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r, held := overloadedRig()
		defer r.Close()

		_, err := r.Server.Detect(context.Background(), &pb.DetectRequest{Image: testrig.Image(99)})
		if status.Code(err) != codes.ResourceExhausted {
			t.Fatalf("code = %v, want RESOURCE_EXHAUSTED", status.Code(err))
		}
		close(r.Worker.Block)
		waitAll(held...)
		synctest.Wait()

		var ids []uint64
		for _, c := range r.Worker.Calls() {
			ids = append(ids, c.IDs...)
		}
		// The rejected request was the 6th to arrive, so it drew ID 6.
		if !slices.Equal(ids, []uint64{1, 2, 3, 4, 5}) {
			t.Fatalf("worker saw IDs %v, want exactly [1 2 3 4 5] (6 was rejected)", ids)
		}
		for _, c := range held {
			if c.Err != nil {
				t.Errorf("admitted request failed: %v", c.Err)
			}
		}
	})
}

func TestDetect_QueueRecoversAfterDrain(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r, held := overloadedRig()
		defer r.Close()
		if _, err := r.Server.Detect(context.Background(), &pb.DetectRequest{Image: testrig.Image(99)}); status.Code(err) != codes.ResourceExhausted {
			t.Fatalf("setup: want overflow, got %v", err)
		}

		close(r.Worker.Block)
		waitAll(held...)

		again := r.Go(context.Background(), testrig.Image(7))
		<-again.Done
		if again.Err != nil {
			t.Fatalf("after drain: %v, want success", again.Err)
		}
	})
}

func TestDetect_EmptyImageInvalidArgument(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 1}})
		defer r.Close()

		for _, img := range [][]byte{nil, {}} {
			_, err := r.Server.Detect(context.Background(), &pb.DetectRequest{Image: img})
			if got := status.Code(err); got != codes.InvalidArgument {
				t.Errorf("code = %v, want INVALID_ARGUMENT", got)
			}
		}
		synctest.Wait()
		if n := len(r.Worker.Calls()); n != 0 {
			t.Errorf("worker saw %d calls: an invalid request must not be queued", n)
		}
		if got := testutil.ToFloat64(r.Metrics.Rejected.WithLabelValues(metrics.RejectInvalid)); got != 2 {
			t.Errorf("rejected{invalid} = %v, want 2", got)
		}
	})
}

func TestDetect_OversizeImageInvalidArgument(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 1}, MaxImageBytes: 100})
		defer r.Close()

		_, err := r.Server.Detect(context.Background(), &pb.DetectRequest{Image: testrig.Image(101)})
		if got := status.Code(err); got != codes.InvalidArgument {
			t.Errorf("101 bytes: code = %v, want INVALID_ARGUMENT", got)
		}
		if _, err := r.Server.Detect(context.Background(), &pb.DetectRequest{Image: testrig.Image(100)}); err != nil {
			t.Errorf("100 bytes (exactly the limit) should be accepted: %v", err)
		}
	})
}
