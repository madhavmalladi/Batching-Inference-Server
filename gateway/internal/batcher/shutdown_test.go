package batcher_test

import (
	"context"
	"testing"
	"testing/synctest"

	"batchserve/gateway/internal/batcher"
	"batchserve/gateway/internal/testrig"

	"google.golang.org/grpc/codes"
)

func TestBatcher_ShutdownFailsQueuedFinishesInflightRejectsNew(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 8, MaxWait: ms, MaxInflight: 1}})
		defer r.Close()
		inflight := holdWorker(r)
		queued := submit(r, 2)

		r.Stop()
		synctest.Wait()

		for i, c := range queued {
			select {
			case <-c.Done:
				if c.Code() != codes.Unavailable {
					t.Errorf("queued %d: code = %v, want UNAVAILABLE", i, c.Code())
				}
			default:
				t.Errorf("queued %d is still waiting: shutdown must fail queued requests promptly", i)
			}
		}
		select {
		case <-inflight.Done:
			t.Fatal("the in-flight request finished before the worker was released")
		default:
		}

		late := r.Go(context.Background(), testrig.Image(5))
		<-late.Done
		if late.Code() != codes.Unavailable {
			t.Errorf("new request during shutdown: code = %v, want UNAVAILABLE", late.Code())
		}

		close(r.Worker.Block)
		<-inflight.Done
		if inflight.Err != nil {
			t.Errorf("in-flight request: %v, want it to complete normally", inflight.Err)
		}
	})
}
