package batcher_test

import (
	"context"
	"slices"
	"testing"
	"testing/synctest"

	"batchserve/gateway/internal/batcher"
	"batchserve/gateway/internal/metrics"
	"batchserve/gateway/internal/testrig"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"google.golang.org/grpc/codes"
)

func expired(r *testrig.Rig, stage string) float64 {
	return testutil.ToFloat64(r.Metrics.Expired.WithLabelValues(stage))
}

// holdWorker parks one request at the worker (F=1, so the batcher can't send
// anything else) and returns it. Callers release the worker with close(r.Worker.Block).
func holdWorker(r *testrig.Rig) *testrig.Call {
	r.Worker.Block = make(chan struct{})
	c := r.Go(context.Background(), testrig.Image(5))
	sleepUntil(r, 2*ms) // W=1ms: dispatched at 1ms and now parked in the worker
	return c
}

func TestBatcher_DropsExpiredRequestsBeforeDispatch(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 8, MaxWait: 1 * ms, MaxInflight: 1}})
		defer r.Close()
		blocker := holdWorker(r)

		ctxX, cancel := context.WithTimeout(context.Background(), 5*ms) // expires at t=7ms
		defer cancel()
		x := r.Go(ctxX, testrig.Image(6))
		synctest.Wait()
		y := r.Go(context.Background(), testrig.Image(7))
		synctest.Wait()

		sleepUntil(r, 20*ms)
		close(r.Worker.Block)
		waitAll([]*testrig.Call{blocker, x, y})

		calls := r.Worker.Calls()
		if len(calls) != 2 || !slices.Equal(calls[1].IDs, []uint64{3}) {
			t.Fatalf("worker calls = %+v, want the second batch to be {Y} only (id 3)", calls)
		}
		if x.Code() != codes.DeadlineExceeded || x.At != 7*ms {
			t.Errorf("X: code=%v at=%v, want DEADLINE_EXCEEDED at 7ms", x.Code(), x.At)
		}
		if y.Err != nil {
			t.Errorf("Y failed: %v", y.Err)
		}
		if got := expired(r, metrics.StageQueued); got != 1 {
			t.Errorf("expired_total{stage=queued} = %v, want 1", got)
		}
	})
}

func TestBatcher_AllExpiredBatchIsNotSent(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 8, MaxWait: 1 * ms, MaxInflight: 1}})
		defer r.Close()
		blocker := holdWorker(r)

		var dead []*testrig.Call
		for range 2 {
			ctx, cancel := context.WithTimeout(context.Background(), 5*ms)
			defer cancel()
			dead = append(dead, r.Go(ctx, testrig.Image(6)))
			synctest.Wait()
		}
		sleepUntil(r, 20*ms)
		close(r.Worker.Block)
		waitAll(append(dead, blocker))
		synctest.Wait()

		if n := len(r.Worker.Calls()); n != 1 {
			t.Fatalf("worker got %d calls, want 1 (only the blocker): an all-expired batch must not be sent", n)
		}
		if got := expired(r, metrics.StageQueued); got != 2 {
			t.Errorf("expired_total{stage=queued} = %v, want 2", got)
		}

		// The slot must not have leaked: a fresh request still goes through.
		fresh := r.Go(context.Background(), testrig.Image(8))
		<-fresh.Done
		if fresh.Err != nil || len(r.Worker.Calls()) != 2 {
			t.Fatalf("fresh request: err=%v calls=%d; want success and 2 worker calls", fresh.Err, len(r.Worker.Calls()))
		}
	})
}

func TestBatcher_ExpiryWhileBatchIsFormingIsDropped(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 8, MaxWait: 10 * ms}})
		defer r.Close()

		// Both requests die (3ms) while the batcher is still waiting out W=10ms.
		var calls []*testrig.Call
		for range 2 {
			ctx, cancel := context.WithTimeout(context.Background(), 3*ms)
			defer cancel()
			calls = append(calls, r.Go(ctx, testrig.Image(5)))
		}
		waitAll(calls)
		sleepUntil(r, 15*ms)

		if n := len(r.Worker.Calls()); n != 0 {
			t.Fatalf("worker got %d calls, want 0: requests that expired inside the forming batch must be dropped at dispatch", n)
		}
		if got := expired(r, metrics.StageQueued); got != 2 {
			t.Errorf("expired_total{stage=queued} = %v, want 2", got)
		}
		ok := r.Go(context.Background(), testrig.Image(5))
		<-ok.Done
		if ok.Err != nil {
			t.Fatalf("slot leaked after an all-expired batch: %v", ok.Err)
		}
	})
}
