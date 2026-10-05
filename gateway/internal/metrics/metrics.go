// Package metrics defines the gateway's Prometheus collectors. The metric names
// and labels are a contract (see CLAUDE.md): sweep scripts and PromQL depend on them.
package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc/codes"
)

// Label values, shared so a typo can't silently create a new time series.
const (
	ReasonFull    = "full"
	ReasonTimeout = "timeout"

	RejectQueueFull = "queue_full"
	RejectInvalid   = "invalid"

	StageAdmission = "admission"
	StageQueued    = "queued"
	StageInflight  = "inflight"
)

// Metrics is a struct of collectors built against an injected Registerer, so
// each test gets a fresh registry instead of sharing global state.
type Metrics struct {
	reg prometheus.Registerer

	BatchSize       prometheus.Histogram
	BatchesTotal    *prometheus.CounterVec   // reason
	RequestDuration *prometheus.HistogramVec // code
	QueueWait       prometheus.Histogram
	BatchDuration   prometheus.Histogram
	WorkerInference prometheus.Histogram
	Rejected        *prometheus.CounterVec // reason
	Expired         *prometheus.CounterVec // stage
}

// Latency buckets: exponential from 0.5 ms, doubling 16 times, so the top
// finite bucket is ~16 s. That covers the default 10 s worker timeout; a
// finer top end would only matter if we tuned timeouts above that.
var latencyBuckets = prometheus.ExponentialBuckets(0.0005, 2, 16)

var batchSizeBuckets = []float64{1, 2, 4, 8, 16, 32, 64}

// codeNames are the gRPC codes the gateway can return. Label values use the
// proto/spec spelling (RESOURCE_EXHAUSTED), not grpc-go's CamelCase String().
var codeNames = map[codes.Code]string{
	codes.OK:                "OK",
	codes.InvalidArgument:   "INVALID_ARGUMENT",
	codes.DeadlineExceeded:  "DEADLINE_EXCEEDED",
	codes.Canceled:          "CANCELLED",
	codes.ResourceExhausted: "RESOURCE_EXHAUSTED",
	codes.Internal:          "INTERNAL",
	codes.Unavailable:       "UNAVAILABLE",
	codes.Unknown:           "UNKNOWN",
}

// CodeName converts a gRPC code to the label value used in request_duration.
func CodeName(c codes.Code) string {
	if n, ok := codeNames[c]; ok {
		return n
	}
	return "UNKNOWN"
}

func New(reg prometheus.Registerer) *Metrics {
	f := func(name, help string, buckets []float64) prometheus.Histogram {
		return prometheus.NewHistogram(prometheus.HistogramOpts{Name: name, Help: help, Buckets: buckets})
	}
	m := &Metrics{
		reg:       reg,
		BatchSize: f("batchserve_batch_size", "Images per dispatched batch.", batchSizeBuckets),
		BatchesTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "batchserve_batches_total", Help: "Batches dispatched, by why they were sent.",
		}, []string{"reason"}),
		RequestDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name: "batchserve_request_duration_seconds", Help: "Admission to response, by gRPC code.",
			Buckets: latencyBuckets,
		}, []string{"code"}),
		QueueWait:       f("batchserve_queue_wait_seconds", "Admission to batch dispatch.", latencyBuckets),
		BatchDuration:   f("batchserve_batch_duration_seconds", "InferBatch round trip.", latencyBuckets),
		WorkerInference: f("batchserve_worker_inference_seconds", "Worker-reported model run time per batch.", latencyBuckets),
		Rejected: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "batchserve_requests_rejected_total", Help: "Requests refused at admission.",
		}, []string{"reason"}),
		Expired: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "batchserve_requests_expired_total", Help: "Requests whose deadline passed or that were cancelled.",
		}, []string{"stage"}),
	}
	reg.MustRegister(m.BatchSize, m.BatchesTotal, m.RequestDuration, m.QueueWait,
		m.BatchDuration, m.WorkerInference, m.Rejected, m.Expired)

	// A labelled collector exports nothing until a child exists, so `curl
	// /metrics` would hide it and rate() queries would see a series appear
	// mid-run. Create every known child up front so they all start at zero.
	for _, r := range []string{ReasonFull, ReasonTimeout} {
		m.BatchesTotal.WithLabelValues(r)
	}
	for _, r := range []string{RejectQueueFull, RejectInvalid} {
		m.Rejected.WithLabelValues(r)
	}
	for _, s := range []string{StageAdmission, StageQueued, StageInflight} {
		m.Expired.WithLabelValues(s)
	}
	for _, n := range codeNames {
		m.RequestDuration.WithLabelValues(n)
	}
	return m
}

// BindQueueDepth registers batchserve_queue_depth as a GaugeFunc that calls fn
// at scrape time. The queue is a Go channel, and len(ch) is always exact, so
// reading it on demand can never drift the way a gauge that we Inc()/Dec() on
// every enqueue/dequeue could. The batcher owns the channel, so it calls this.
func (m *Metrics) BindQueueDepth(fn func() float64) {
	m.reg.MustRegister(prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Name: "batchserve_queue_depth", Help: "Requests waiting in the admission queue.",
	}, fn))
}

// Handler serves /metrics for the given registry (a Gatherer).
func Handler(g prometheus.Gatherer) http.Handler {
	return promhttp.HandlerFor(g, promhttp.HandlerOpts{})
}
