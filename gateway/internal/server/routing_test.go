package server_test

import (
	"context"
	"testing"
	"testing/synctest"
	"time"

	"batchserve/gateway/internal/batcher"
	"batchserve/gateway/internal/fakeworker"
	"batchserve/gateway/internal/testrig"
	pb "batchserve/gen/go/batchservev1"

	"google.golang.org/grpc/codes"
)

// stubWorker lets a test return a deliberately malformed response.
type stubWorker func(context.Context, *pb.InferBatchRequest) (*pb.InferBatchResponse, error)

func (f stubWorker) InferBatch(ctx context.Context, req *pb.InferBatchRequest) (*pb.InferBatchResponse, error) {
	return f(ctx, req)
}

// goodResponse is what a correct worker would answer, produced by the fake.
func goodResponse(req *pb.InferBatchRequest) *pb.InferBatchResponse {
	resp, err := fakeworker.New(fakeworker.Config{}).InferBatch(context.Background(), req)
	if err != nil {
		panic(err)
	}
	return resp
}

func TestRouting_EachClientGetsOwnResult(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{
			Batcher: batcher.Config{MaxBatchSize: 16, MaxWait: time.Millisecond, MaxInflight: 2},
			Worker:  fakeworker.Config{BaseLatency: 3 * time.Millisecond, PerImageLatency: time.Millisecond},
		})
		defer r.Close()

		const n = 200
		calls := make([]*testrig.Call, n)
		for i := range calls {
			// Each payload has a distinct length (1..200); the fake echoes the
			// length back as box.x_max, so a misrouted result is detectable.
			calls[i] = r.Go(context.Background(), testrig.Image(i+1))
		}
		waitAll(calls...)

		for i, c := range calls {
			if c.Err != nil {
				t.Fatalf("request %d: %v", i, c.Err)
			}
			if len(c.Resp.Detections) != 1 {
				t.Fatalf("request %d: %d detections, want 1", i, len(c.Resp.Detections))
			}
			if got, want := c.Resp.Detections[0].Box.XMax, float32(i+1); got != want {
				t.Fatalf("request %d got x_max=%v, want %v: result routed to the wrong client", i, got, want)
			}
		}
	})
}

func TestRouting_PerImageErrorFailsOnlyThatRequest(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 4, MaxWait: time.Millisecond}})
		defer r.Close()

		images := [][]byte{testrig.Image(5), []byte("BADIMG-not-a-jpeg"), testrig.Image(7), testrig.Image(8)}
		var calls []*testrig.Call
		for _, img := range images {
			calls = append(calls, r.Go(context.Background(), img))
			synctest.Wait()
		}
		waitAll(calls...)

		if n := len(r.Worker.Calls()); n != 1 {
			t.Fatalf("worker calls = %d, want the 4 requests in one batch", n)
		}
		for i, c := range calls {
			want := codes.OK
			if i == 1 {
				want = codes.InvalidArgument
			}
			if c.Code() != want {
				t.Errorf("request %d: code = %v, want %v", i, c.Code(), want)
			}
		}
	})
}

func TestRouting_WorkerErrorFailsWholeBatch(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{
			Batcher: batcher.Config{MaxBatchSize: 4, MaxWait: time.Millisecond},
			Worker:  fakeworker.Config{FailRate: 1},
		})
		defer r.Close()

		var calls []*testrig.Call
		for range 4 {
			calls = append(calls, r.Go(context.Background(), testrig.Image(5)))
		}
		waitAll(calls...)
		for i, c := range calls {
			if c.Code() != codes.Unavailable {
				t.Errorf("request %d: code = %v, want UNAVAILABLE", i, c.Code())
			}
		}
	})
}

func runWithStub(t *testing.T, stub stubWorker) []*testrig.Call {
	r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 4, MaxWait: time.Millisecond}, Inferer: stub})
	defer r.Close()
	var calls []*testrig.Call
	for i := range 4 {
		calls = append(calls, r.Go(context.Background(), testrig.Image(5+i)))
	}
	waitAll(calls...)
	return calls
}

func TestRouting_ResultCountMismatchIsInternal(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		calls := runWithStub(t, func(_ context.Context, req *pb.InferBatchRequest) (*pb.InferBatchResponse, error) {
			resp := goodResponse(req)
			resp.Results = resp.Results[:len(resp.Results)-1] // N-1 results
			return resp, nil
		})
		for i, c := range calls {
			if c.Code() != codes.Internal {
				t.Errorf("request %d: code = %v, want INTERNAL", i, c.Code())
			}
		}
	})
}

func TestRouting_RequestIDMismatchIsInternal(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		calls := runWithStub(t, func(_ context.Context, req *pb.InferBatchRequest) (*pb.InferBatchResponse, error) {
			resp := goodResponse(req)
			// Swap the IDs of the first two results: right count, wrong routing.
			resp.Results[0].RequestId, resp.Results[1].RequestId = resp.Results[1].RequestId, resp.Results[0].RequestId
			return resp, nil
		})
		for i, c := range calls {
			if c.Code() != codes.Internal {
				t.Errorf("request %d: code = %v, want INTERNAL for every request: no silent misrouting", i, c.Code())
			}
			if c.Resp != nil {
				t.Errorf("request %d got a response despite the ID mismatch", i)
			}
		}
	})
}

func TestDetect_ResponseIncludesServerTiming(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{
			Batcher: batcher.Config{MaxBatchSize: 4, MaxWait: 10 * time.Millisecond},
			Worker:  fakeworker.Config{BaseLatency: 5 * time.Millisecond, PerImageLatency: 2 * time.Millisecond},
		})
		defer r.Close()

		// One lonely request: waits the full W=10ms, then a batch of 1 takes 5+2=7ms.
		lone := r.Go(context.Background(), testrig.Image(5))
		<-lone.Done
		got := lone.Resp.Timing
		if got.BatchSize != 1 || got.QueueWaitUs != 10_000 || got.BatchLatencyUs != 7_000 ||
			got.DispatchReason != pb.DispatchReason_DISPATCH_REASON_TIMEOUT {
			t.Errorf("lone timing = %v, want batch_size=1 queue_wait=10000us batch_latency=7000us TIMEOUT", got)
		}

		// A full batch: no queue wait, 5+4×2 = 13ms at the worker.
		var full []*testrig.Call
		for range 4 {
			full = append(full, r.Go(context.Background(), testrig.Image(5)))
		}
		waitAll(full...)
		got = full[0].Resp.Timing
		if got.BatchSize != 4 || got.QueueWaitUs != 0 || got.BatchLatencyUs != 13_000 ||
			got.DispatchReason != pb.DispatchReason_DISPATCH_REASON_FULL {
			t.Errorf("full timing = %v, want batch_size=4 queue_wait=0 batch_latency=13000us FULL", got)
		}
	})
}
