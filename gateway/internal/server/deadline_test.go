package server_test

import (
	"context"
	"testing"
	"testing/synctest"
	"time"

	"batchserve/gateway/internal/batcher"
	"batchserve/gateway/internal/fakeworker"
	"batchserve/gateway/internal/metrics"
	"batchserve/gateway/internal/testrig"
	pb "batchserve/gen/go/batchservev1"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const ms = time.Millisecond

func expired(r *testrig.Rig, stage string) float64 {
	return testutil.ToFloat64(r.Metrics.Expired.WithLabelValues(stage))
}

func TestDetect_ExpiredDeadlineRejectedAtAdmission(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 4, MaxWait: 5 * ms}})
		defer r.Close()

		ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Millisecond))
		defer cancel()
		_, err := r.Server.Detect(ctx, &pb.DetectRequest{Image: testrig.Image(5)})

		if got := status.Code(err); got != codes.DeadlineExceeded {
			t.Fatalf("code = %v, want DEADLINE_EXCEEDED", got)
		}
		if got := expired(r, metrics.StageAdmission); got != 1 {
			t.Errorf("expired_total{stage=admission} = %v, want 1", got)
		}
		if got := testutil.ToFloat64(r.Metrics.Rejected.WithLabelValues(metrics.RejectQueueFull)); got != 0 {
			t.Errorf("queue_full = %v, want 0 (it was never queued)", got)
		}
		synctest.Wait()
		if n := len(r.Worker.Calls()); n != 0 {
			t.Errorf("worker saw %d calls, want 0", n)
		}
	})
}

func TestDetect_DeadlineDuringInferenceReturnsAtDeadline(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{
			Batcher: batcher.Config{MaxBatchSize: 2, MaxWait: 10 * ms},
			Worker:  fakeworker.Config{BaseLatency: 100 * ms},
		})
		defer r.Close()

		impatient, cancel := context.WithTimeout(context.Background(), 30*ms)
		defer cancel()
		x := r.Go(impatient, testrig.Image(5))
		y := r.Go(context.Background(), testrig.Image(6)) // same batch (B=2, FULL at t=0)
		waitAll(x, y)

		if x.Code() != codes.DeadlineExceeded || x.At != 30*ms {
			t.Errorf("X: code=%v at=%v, want DEADLINE_EXCEEDED at 30ms (not 100ms)", x.Code(), x.At)
		}
		if y.Err != nil || y.At != 100*ms {
			t.Errorf("Y: err=%v at=%v, want OK at 100ms: one client's deadline must not hurt its batch-mates", y.Err, y.At)
		}
		if got := expired(r, metrics.StageInflight); got != 1 {
			t.Errorf("expired_total{stage=inflight} = %v, want 1", got)
		}
	})
}

func TestDetect_LateResultDoesNotLeakGoroutine(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{
			Batcher: batcher.Config{MaxBatchSize: 2, MaxWait: 10 * ms},
			Worker:  fakeworker.Config{BaseLatency: 100 * ms},
		})

		impatient, cancel := context.WithTimeout(context.Background(), 30*ms)
		defer cancel()
		x := r.Go(impatient, testrig.Image(5))
		y := r.Go(context.Background(), testrig.Image(6))
		waitAll(x, y)
		synctest.Wait()

		// X's reply channel received a result nobody will read. If that send
		// could block (unbuffered channel), the dispatch goroutine would be
		// stuck forever and Close() below would hang until synctest reports a
		// deadlock. Close returning is the leak check.
		r.Close()
	})
}

func TestDetect_ClientCancelReturnsCanceled(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 8, MaxWait: 1 * ms, MaxInflight: 1}})
		defer r.Close()
		r.Worker.Block = make(chan struct{})
		blocker := r.Go(context.Background(), testrig.Image(5))
		<-time.After(2 * ms) // dispatched at 1ms, parked in the worker

		ctx, cancel := context.WithCancel(context.Background())
		x := r.Go(ctx, testrig.Image(6))
		synctest.Wait()
		<-time.After(3 * ms)
		cancel()
		<-x.Done
		if x.Code() != codes.Canceled || x.At != 5*ms {
			t.Fatalf("X: code=%v at=%v, want CANCELED at 5ms (when it cancelled, not when the worker freed up)", x.Code(), x.At)
		}

		<-time.After(10 * ms)
		close(r.Worker.Block)
		<-blocker.Done
		synctest.Wait()

		if n := len(r.Worker.Calls()); n != 1 {
			t.Errorf("worker got %d calls, want 1: the cancelled request must be skipped at dispatch", n)
		}
		if got := expired(r, metrics.StageQueued); got != 1 {
			t.Errorf("expired_total{stage=queued} = %v, want 1", got)
		}
	})
}

func waitAll(calls ...*testrig.Call) {
	for _, c := range calls {
		<-c.Done
	}
}
