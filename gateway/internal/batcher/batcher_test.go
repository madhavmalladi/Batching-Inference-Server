package batcher_test

// Batching-policy tests. Every test runs in a testing/synctest bubble, where
// time.Now() is virtual: time only moves when every goroutine is blocked, so
// "dispatched at exactly t=10ms" is an exact assertion and the test runs
// instantly. The fake worker has zero latency unless a test says otherwise, so
// the time a batch reaches the worker IS the time the batcher dispatched it.

import (
	"context"
	"slices"
	"testing"
	"testing/synctest"
	"time"

	pb "batchserve/gen/go/batchservev1"
	"batchserve/gateway/internal/batcher"
	"batchserve/gateway/internal/fakeworker"
	"batchserve/gateway/internal/testrig"
)

const ms = time.Millisecond

// sleepUntil advances virtual time to offset d from the rig's start. (It waits
// on a timer rather than calling time.Sleep: same effect inside a synctest
// bubble, and it keeps real sleeps out of the test suite.)
func sleepUntil(r *testrig.Rig, d time.Duration) {
	<-time.After(d - r.Elapsed())
}

// submit starts n requests, waiting for each to be admitted before the next so
// their IDs and queue order are deterministic.
func submit(r *testrig.Rig, n int) []*testrig.Call {
	calls := make([]*testrig.Call, n)
	for i := range calls {
		calls[i] = r.Go(context.Background(), testrig.Image(10+i))
		synctest.Wait()
	}
	return calls
}

func waitAll(calls []*testrig.Call) {
	for _, c := range calls {
		<-c.Done
	}
}

// arrivals summarizes worker calls as (size, time since start) pairs.
type arrival struct {
	size int
	at   time.Duration
}

func arrivals(r *testrig.Rig) []arrival {
	var out []arrival
	for _, c := range r.Worker.Calls() {
		out = append(out, arrival{c.Size, c.At.Sub(r.Start)})
	}
	return out
}

func reasonOf(t *testing.T, c *testrig.Call) pb.DispatchReason {
	t.Helper()
	if c.Err != nil {
		t.Fatalf("request failed: %v", c.Err)
	}
	return c.Resp.Timing.DispatchReason
}

func TestBatcher_SingleRequestTimesOutAlone(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 8, MaxWait: 10 * ms}})
		defer r.Close()

		c := submit(r, 1)[0]
		<-c.Done

		if got := arrivals(r); !slices.Equal(got, []arrival{{1, 10 * ms}}) {
			t.Fatalf("arrivals = %v, want one batch of 1 at 10ms", got)
		}
		if got := reasonOf(t, c); got != pb.DispatchReason_DISPATCH_REASON_TIMEOUT {
			t.Errorf("reason = %v, want TIMEOUT", got)
		}
	})
}

func TestBatcher_BurstFillsBatchImmediately(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 8, MaxWait: 10 * ms}})
		defer r.Close()

		calls := submit(r, 8)
		waitAll(calls)

		if got := arrivals(r); !slices.Equal(got, []arrival{{8, 0}}) {
			t.Fatalf("arrivals = %v, want one batch of 8 at t=0 (no wait)", got)
		}
		if got := reasonOf(t, calls[0]); got != pb.DispatchReason_DISPATCH_REASON_FULL {
			t.Errorf("reason = %v, want FULL", got)
		}
	})
}

func TestBatcher_BurstLargerThanBatchSplits(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 8, MaxWait: 10 * ms, MaxInflight: 3}})
		defer r.Close()

		calls := submit(r, 20)
		waitAll(calls)

		// The two full batches may reach the worker in either order (they run
		// in separate goroutines), so compare as a sorted set.
		got := arrivals(r)
		slices.SortFunc(got, func(a, b arrival) int { return int(a.at-b.at)*100 + (b.size - a.size) })
		want := []arrival{{8, 0}, {8, 0}, {4, 10 * ms}}
		if !slices.Equal(got, want) {
			t.Fatalf("arrivals = %v, want %v (the remainder waits W from ITS admission, t=0)", got, want)
		}
		full, timeout := 0, 0
		for _, c := range calls {
			switch reasonOf(t, c) {
			case pb.DispatchReason_DISPATCH_REASON_FULL:
				full++
			case pb.DispatchReason_DISPATCH_REASON_TIMEOUT:
				timeout++
			}
		}
		if full != 16 || timeout != 4 {
			t.Errorf("reasons: %d full, %d timeout; want 16 and 4", full, timeout)
		}
	})
}

func TestBatcher_TimerStartsAtOldestRequest(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 8, MaxWait: 10 * ms}})
		defer r.Close()

		c1 := r.Go(context.Background(), testrig.Image(5))
		sleepUntil(r, 8*ms)
		c2 := r.Go(context.Background(), testrig.Image(6))
		waitAll([]*testrig.Call{c1, c2})

		if got := arrivals(r); !slices.Equal(got, []arrival{{2, 10 * ms}}) {
			t.Fatalf("arrivals = %v, want one batch of 2 at 10ms (not 18ms)", got)
		}
	})
}

func TestBatcher_BacklogDispatchesWithoutWaiting(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 8, MaxWait: 10 * ms, MaxInflight: 1}})
		defer r.Close()
		r.Worker.Block = make(chan struct{})

		first := r.Go(context.Background(), testrig.Image(5)) // occupies the only slot
		sleepUntil(r, 11*ms)                                  // it timed out at 10ms and is now stuck in the worker
		backlog := submit(r, 3)
		sleepUntil(r, 50*ms)
		close(r.Worker.Block)
		waitAll(append(backlog, first))

		want := []arrival{{1, 10 * ms}, {3, 50 * ms}}
		if got := arrivals(r); !slices.Equal(got, want) {
			t.Fatalf("arrivals = %v, want %v: the backlog's oldest request was already past its deadline, so no extra wait", got, want)
		}
	})
}

func TestBatcher_MaxWaitZeroIsGreedy(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 8, MaxWait: 0, MaxInflight: 1}})
		defer r.Close()
		r.Worker.Block = make(chan struct{})

		first := r.Go(context.Background(), testrig.Image(5))
		synctest.Wait() // dispatched alone at t=0, now parked in the worker
		queued := submit(r, 3)
		sleepUntil(r, 5*ms)
		close(r.Worker.Block)
		waitAll(append(queued, first))
		sleepUntil(r, 20*ms)
		late := r.Go(context.Background(), testrig.Image(7))
		<-late.Done

		want := []arrival{{1, 0}, {3, 5 * ms}, {1, 20 * ms}}
		if got := arrivals(r); !slices.Equal(got, want) {
			t.Fatalf("arrivals = %v, want %v: each batch is exactly what was queued, with no waiting", got, want)
		}
	})
}

func TestBatcher_MaxBatchSizeOneNeverWaits(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 1, MaxWait: 100 * ms}})
		defer r.Close()

		calls := submit(r, 3)
		waitAll(calls)

		want := []arrival{{1, 0}, {1, 0}, {1, 0}}
		if got := arrivals(r); !slices.Equal(got, want) {
			t.Fatalf("arrivals = %v, want %v (B=1 is the no-batching baseline)", got, want)
		}
		if got := reasonOf(t, calls[0]); got != pb.DispatchReason_DISPATCH_REASON_FULL {
			t.Errorf("reason = %v, want FULL", got)
		}
	})
}

func TestBatcher_PreservesFIFOOrder(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		// F=1 so batches reach the worker strictly one after another; with F>1
		// two dispatch goroutines could race to call the worker.
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 4, MaxWait: 5 * ms, MaxInflight: 1}})
		defer r.Close()

		waitAll(submit(r, 10))

		var ids []uint64
		for _, c := range r.Worker.Calls() {
			ids = append(ids, c.IDs...)
		}
		want := []uint64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		if !slices.Equal(ids, want) {
			t.Fatalf("worker saw IDs %v, want admission order %v", ids, want)
		}
	})
}

func TestBatcher_RespectsMaxInflight(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{
			Batcher: batcher.Config{MaxBatchSize: 4, MaxWait: 1 * ms, MaxInflight: 2},
			Worker:  fakeworker.Config{BaseLatency: 10 * ms},
		})
		defer r.Close()

		waitAll(submit(r, 50))

		// <= 2 is the contract; == 2 shows the test could have caught a limit
		// that was too strict (F=1 behaviour).
		if got := r.Worker.MaxConcurrent(); got != 2 {
			t.Fatalf("MaxConcurrent = %d, want exactly 2", got)
		}
	})
}

func TestBatcher_AcquiresSlotBeforeFormingBatch(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{
			Batcher: batcher.Config{MaxBatchSize: 8, MaxWait: 10 * ms, MaxInflight: 1},
			Worker:  fakeworker.Config{BaseLatency: 20 * ms},
		})
		defer r.Close()

		busy := submit(r, 8) // a full batch: dispatched at t=0, worker busy until t=20ms
		var late []*testrig.Call
		for _, at := range []time.Duration{1 * ms, 5 * ms, 10 * ms, 15 * ms} {
			sleepUntil(r, at)
			late = append(late, r.Go(context.Background(), testrig.Image(3)))
		}
		waitAll(append(busy, late...))

		// If the batcher had formed a batch first and THEN waited for a slot,
		// it would have closed the batch at t=11ms with only 3 requests in it.
		want := []arrival{{8, 0}, {4, 20 * ms}}
		if got := arrivals(r); !slices.Equal(got, want) {
			t.Fatalf("arrivals = %v, want %v: requests arriving while the worker is busy must all join the next batch", got, want)
		}
	})
}
