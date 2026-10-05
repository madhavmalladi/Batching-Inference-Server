package fakeworker_test

import (
	"context"
	"hash/crc32"
	"testing"
	"testing/synctest"
	"time"

	"batchserve/gateway/internal/fakeworker"
	pb "batchserve/gen/go/batchservev1"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func batchOf(datas ...[]byte) *pb.InferBatchRequest {
	req := &pb.InferBatchRequest{BatchId: 1}
	for i, d := range datas {
		req.Images = append(req.Images, &pb.WorkerImage{RequestId: uint64(i + 1), Data: d})
	}
	return req
}

func goodImages(n int) [][]byte {
	out := make([][]byte, n)
	for i := range out {
		out[i] = []byte("image-bytes")
	}
	return out
}

// All timing tests run inside a synctest bubble: time.Now() is virtual and only
// advances when every goroutine is blocked, so "7ms" is exact and the test is instant.
func TestFakeWorker_LatencyScalesWithBatchSize(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		w := fakeworker.New(fakeworker.Config{
			BaseLatency:     5 * time.Millisecond,
			PerImageLatency: 2 * time.Millisecond,
		})
		for _, tc := range []struct {
			n    int
			want time.Duration
		}{{1, 7 * time.Millisecond}, {8, 21 * time.Millisecond}} {
			start := time.Now()
			if _, err := w.InferBatch(context.Background(), batchOf(goodImages(tc.n)...)); err != nil {
				t.Fatalf("N=%d: %v", tc.n, err)
			}
			if got := time.Since(start); got != tc.want {
				t.Errorf("N=%d slept %v, want %v", tc.n, got, tc.want)
			}
		}
	})
}

func TestFakeWorker_DeterministicDetections(t *testing.T) {
	w := fakeworker.New(fakeworker.Config{})
	data := []byte("hello")
	req := batchOf(data, data, []byte("a different, longer payload"))
	resp, err := w.InferBatch(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Results) != 3 {
		t.Fatalf("got %d results, want 3", len(resp.Results))
	}
	for i, r := range resp.Results {
		if r.RequestId != req.Images[i].RequestId {
			t.Errorf("result %d: request_id %d, want %d", i, r.RequestId, req.Images[i].RequestId)
		}
		if len(r.Detections) != 1 {
			t.Fatalf("result %d: %d detections, want 1", i, len(r.Detections))
		}
		d := r.Detections[0]
		wantClass := int32(crc32.ChecksumIEEE(req.Images[i].Data) % 80)
		if d.ClassId != wantClass {
			t.Errorf("result %d: class_id %d, want %d", i, d.ClassId, wantClass)
		}
		if d.Score != 0.5 {
			t.Errorf("result %d: score %v, want 0.5", i, d.Score)
		}
		if d.Box.XMax != float32(len(req.Images[i].Data)) || d.Box.YMax != d.Box.XMax || d.Box.XMin != 0 || d.Box.YMin != 0 {
			t.Errorf("result %d: box %v, want 0,0,%d,%d", i, d.Box, len(req.Images[i].Data), len(req.Images[i].Data))
		}
	}
	// Same bytes -> same output, regardless of position in the batch.
	if resp.Results[0].Detections[0].ClassId != resp.Results[1].Detections[0].ClassId {
		t.Error("identical images got different classes")
	}
	if resp.Results[0].Detections[0].ClassName == "" {
		t.Error("class_name is empty")
	}
}

func TestFakeWorker_BadImagePerImageError(t *testing.T) {
	w := fakeworker.New(fakeworker.Config{})
	resp, err := w.InferBatch(context.Background(), batchOf(
		[]byte("fine"), []byte("BADIMG-corrupt"), nil, []byte("also fine")))
	if err != nil {
		t.Fatalf("a bad image must not fail the batch: %v", err)
	}
	wantErr := []bool{false, true, true, false}
	for i, r := range resp.Results {
		if (r.Error != "") != wantErr[i] {
			t.Errorf("result %d: error=%q, want error=%v", i, r.Error, wantErr[i])
		}
		if wantErr[i] && len(r.Detections) != 0 {
			t.Errorf("result %d: failed image still has detections", i)
		}
		if !wantErr[i] && len(r.Detections) != 1 {
			t.Errorf("result %d: healthy image has %d detections", i, len(r.Detections))
		}
	}
}

func TestFakeWorker_FailRate(t *testing.T) {
	run := func(rate float64, seed int64, n int) []bool {
		w := fakeworker.New(fakeworker.Config{FailRate: rate, Seed: seed})
		out := make([]bool, n)
		for i := range out {
			_, err := w.InferBatch(context.Background(), batchOf([]byte("x")))
			if err != nil && status.Code(err) != codes.Unavailable {
				t.Fatalf("failure code = %v, want UNAVAILABLE", status.Code(err))
			}
			out[i] = err != nil
		}
		return out
	}
	for _, failed := range run(1, 1, 50) {
		if !failed {
			t.Fatal("fail-rate=1 must always fail")
		}
	}
	for _, failed := range run(0, 1, 50) {
		if failed {
			t.Fatal("fail-rate=0 must never fail")
		}
	}
	a, b := run(0.5, 42, 100), run(0.5, 42, 100)
	nFail := 0
	for i := range a {
		if a[i] != b[i] {
			t.Fatal("same seed must give the same failure sequence")
		}
		if a[i] {
			nFail++
		}
	}
	if nFail == 0 || nFail == len(a) {
		t.Errorf("fail-rate=0.5 produced %d/%d failures; expected a mix", nFail, len(a))
	}
}

func TestFakeWorker_JitterStaysInRange(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		w := fakeworker.New(fakeworker.Config{BaseLatency: 100 * time.Millisecond, Jitter: 0.1, Seed: 7})
		distinct := map[time.Duration]bool{}
		for range 20 {
			start := time.Now()
			if _, err := w.InferBatch(context.Background(), batchOf([]byte("x"))); err != nil {
				t.Fatal(err)
			}
			d := time.Since(start)
			if d < 90*time.Millisecond || d > 110*time.Millisecond {
				t.Fatalf("slept %v, outside 100ms ±10%%", d)
			}
			distinct[d] = true
		}
		if len(distinct) < 2 {
			t.Error("jitter produced identical sleeps")
		}
	})
}

func TestFakeWorker_HonorsContextCancel(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		w := fakeworker.New(fakeworker.Config{BaseLatency: time.Second})
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
		defer cancel()
		start := time.Now()
		_, err := w.InferBatch(ctx, batchOf([]byte("x")))
		if got := status.Code(err); got != codes.DeadlineExceeded {
			t.Fatalf("code = %v, want DEADLINE_EXCEEDED", got)
		}
		if got := time.Since(start); got != 10*time.Millisecond {
			t.Errorf("returned after %v, want 10ms (not the full 1s sleep)", got)
		}

		ctx2, cancel2 := context.WithCancel(context.Background())
		cancel2()
		if _, err := w.InferBatch(ctx2, batchOf([]byte("x"))); status.Code(err) != codes.Canceled {
			t.Errorf("code = %v, want CANCELED", status.Code(err))
		}
	})
}

func TestFakeWorker_BlockHookHoldsBatchUntilReleased(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		w := fakeworker.New(fakeworker.Config{})
		w.Block = make(chan struct{})
		done := make(chan struct{})
		go func() {
			defer close(done)
			if _, err := w.InferBatch(context.Background(), batchOf([]byte("x"))); err != nil {
				t.Error(err)
			}
		}()
		synctest.Wait()
		select {
		case <-done:
			t.Fatal("InferBatch returned while Block was open")
		default:
		}
		if len(w.Calls()) != 1 {
			t.Fatal("a blocked call must already be recorded, so tests can see the batch arrive")
		}
		close(w.Block)
		<-done
	})
}

func TestFakeWorker_RecordsCallsAndConcurrency(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		w := fakeworker.New(fakeworker.Config{BaseLatency: 10 * time.Millisecond})
		start := time.Now()
		done := make(chan struct{}, 3)
		for range 3 {
			go func() {
				_, _ = w.InferBatch(context.Background(), batchOf([]byte("a"), []byte("b")))
				done <- struct{}{}
			}()
		}
		for range 3 {
			<-done
		}
		calls := w.Calls()
		if len(calls) != 3 {
			t.Fatalf("recorded %d calls, want 3", len(calls))
		}
		for _, c := range calls {
			if c.Size != 2 || len(c.IDs) != 2 || c.IDs[0] != 1 || c.IDs[1] != 2 || !c.At.Equal(start) {
				t.Errorf("unexpected record %+v", c)
			}
		}
		if got := w.MaxConcurrent(); got != 3 {
			t.Errorf("MaxConcurrent = %d, want 3", got)
		}
	})
}
