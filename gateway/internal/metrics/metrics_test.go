package metrics_test

// These tests drive the real batcher and handler (via testrig) and then read
// the registry, so they check what an operator would actually see on /metrics,
// not just that a collector object exists. They live in an external test
// package because testrig imports metrics.

import (
	"context"
	"io"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"batchserve/gateway/internal/batcher"
	"batchserve/gateway/internal/metrics"
	"batchserve/gateway/internal/testrig"
	pb "batchserve/gen/go/batchservev1"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
)

const ms = time.Millisecond

// find returns the metric in family `name` whose labels match want (nil = the only one).
func find(t *testing.T, reg prometheus.Gatherer, name string, want map[string]string) *dto.Metric {
	t.Helper()
	fams, err := reg.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range fams {
		if f.GetName() != name {
			continue
		}
	next:
		for _, m := range f.Metric {
			for k, v := range want {
				found := false
				for _, l := range m.Label {
					if l.GetName() == k && l.GetValue() == v {
						found = true
					}
				}
				if !found {
					continue next
				}
			}
			return m
		}
	}
	t.Fatalf("metric %s%v not found", name, want)
	return nil
}

// burst20 sends 20 requests at once with B=8, W=10ms, F=3: batches of 8, 8, 4.
func burst20(r *testrig.Rig) {
	var calls []*testrig.Call
	for i := range 20 {
		calls = append(calls, r.Go(context.Background(), testrig.Image(i+1)))
		synctest.Wait()
	}
	for _, c := range calls {
		<-c.Done
	}
}

func burstRig() *testrig.Rig {
	return testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 8, MaxWait: 10 * ms, MaxInflight: 3}})
}

func TestMetrics_BatchSizeHistogram(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := burstRig()
		defer r.Close()
		burst20(r)

		h := find(t, r.Registry, "batchserve_batch_size", nil).Histogram
		if h.GetSampleCount() != 3 || h.GetSampleSum() != 20 {
			t.Errorf("batch_size count=%d sum=%v, want count 3 sum 20 (batches of 8, 8, 4)", h.GetSampleCount(), h.GetSampleSum())
		}
		var bounds []float64
		for _, b := range h.Bucket {
			bounds = append(bounds, b.GetUpperBound())
		}
		if len(bounds) != 7 || bounds[0] != 1 || bounds[6] != 64 {
			t.Errorf("batch_size buckets = %v, want 1,2,4,8,16,32,64", bounds)
		}
	})
}

func TestMetrics_BatchesTotalByReason(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := burstRig()
		defer r.Close()
		burst20(r)

		if got := testutil.ToFloat64(r.Metrics.BatchesTotal.WithLabelValues(metrics.ReasonFull)); got != 2 {
			t.Errorf("batches_total{reason=full} = %v, want 2", got)
		}
		if got := testutil.ToFloat64(r.Metrics.BatchesTotal.WithLabelValues(metrics.ReasonTimeout)); got != 1 {
			t.Errorf("batches_total{reason=timeout} = %v, want 1", got)
		}
	})
}

func TestMetrics_RejectedCounter(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 1, QueueSize: 2, MaxInflight: 1}})
		defer r.Close()
		r.Worker.Block = make(chan struct{})
		var held []*testrig.Call
		for i := range 3 { // 1 in flight + 2 queued = full
			held = append(held, r.Go(context.Background(), testrig.Image(i+1)))
			synctest.Wait()
		}

		for range 2 {
			_, _ = r.Server.Detect(context.Background(), &pb.DetectRequest{Image: testrig.Image(9)})
		}
		if got := testutil.ToFloat64(r.Metrics.Rejected.WithLabelValues(metrics.RejectQueueFull)); got != 2 {
			t.Errorf("requests_rejected_total{reason=queue_full} = %v, want 2", got)
		}
		close(r.Worker.Block)
		for _, c := range held {
			<-c.Done
		}
	})
}

func TestMetrics_QueueDepthGauge(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 1, QueueSize: 10, MaxInflight: 1}})
		defer r.Close()
		depth := func() float64 { return find(t, r.Registry, "batchserve_queue_depth", nil).Gauge.GetValue() }

		if depth() != 0 {
			t.Fatalf("idle depth = %v, want 0", depth())
		}
		r.Worker.Block = make(chan struct{})
		var calls []*testrig.Call
		for i := range 4 { // 1 at the worker, 3 waiting in the queue
			calls = append(calls, r.Go(context.Background(), testrig.Image(i+1)))
			synctest.Wait()
		}
		if depth() != 3 {
			t.Errorf("depth = %v, want 3 (the request at the worker is no longer queued)", depth())
		}
		close(r.Worker.Block)
		for _, c := range calls {
			<-c.Done
		}
		if depth() != 0 {
			t.Errorf("drained depth = %v, want 0", depth())
		}
	})
}

func TestMetrics_RequestDurationLabeledByCode(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 1, QueueSize: 1, MaxInflight: 1}})
		defer r.Close()
		r.Worker.Block = make(chan struct{})
		first := r.Go(context.Background(), testrig.Image(1)) // at the worker
		synctest.Wait()
		second := r.Go(context.Background(), testrig.Image(2)) // fills the queue
		synctest.Wait()
		_, _ = r.Server.Detect(context.Background(), &pb.DetectRequest{Image: testrig.Image(3)}) // RESOURCE_EXHAUSTED
		<-time.After(5 * ms)
		close(r.Worker.Block)
		<-first.Done
		<-second.Done

		ok := find(t, r.Registry, "batchserve_request_duration_seconds", map[string]string{"code": "OK"}).Histogram
		re := find(t, r.Registry, "batchserve_request_duration_seconds", map[string]string{"code": "RESOURCE_EXHAUSTED"}).Histogram
		if ok.GetSampleCount() != 2 || re.GetSampleCount() != 1 {
			t.Errorf("counts: OK=%d RESOURCE_EXHAUSTED=%d, want 2 and 1", ok.GetSampleCount(), re.GetSampleCount())
		}
		if re.GetSampleSum() != 0 {
			t.Errorf("a rejection took %vs, want 0 (it is immediate)", re.GetSampleSum())
		}
		if ok.GetSampleSum() < 0.005 {
			t.Errorf("OK requests waited out the 5ms block, but sum = %vs", ok.GetSampleSum())
		}
	})
}

func TestMetrics_EndpointServesAllNames(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := testrig.New(testrig.Options{Batcher: batcher.Config{MaxBatchSize: 1}})
		defer r.Close()

		rec := httptest.NewRecorder()
		metrics.Handler(r.Registry).ServeHTTP(rec, httptest.NewRequest("GET", "/metrics", nil))
		body, _ := io.ReadAll(rec.Result().Body)

		// Straight from the table in CLAUDE.md. This test is the guard against
		// renaming a metric that sweep scripts and dashboards depend on. All of
		// them must be present even before any traffic (labelled children are
		// pre-created for exactly that reason).
		for _, name := range []string{
			"batchserve_queue_depth",
			"batchserve_batch_size",
			"batchserve_batches_total",
			"batchserve_request_duration_seconds",
			"batchserve_queue_wait_seconds",
			"batchserve_batch_duration_seconds",
			"batchserve_worker_inference_seconds",
			"batchserve_requests_rejected_total",
			"batchserve_requests_expired_total",
		} {
			if !strings.Contains(string(body), "\n"+name) && !strings.HasPrefix(string(body), name) {
				t.Errorf("/metrics is missing %s", name)
			}
		}
		for _, series := range []string{
			`batchserve_batches_total{reason="full"} 0`,
			`batchserve_requests_rejected_total{reason="queue_full"} 0`,
			`batchserve_requests_expired_total{stage="inflight"} 0`,
		} {
			if !strings.Contains(string(body), series) {
				t.Errorf("/metrics is missing series %s", series)
			}
		}
	})
}
