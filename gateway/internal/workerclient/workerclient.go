// Package workerclient is the batcher's gRPC connection to the inference worker.
// It is the "real" implementation of batcher.Inferer; the in-process fake is the other.
package workerclient

import (
	"context"
	"fmt"
	"time"

	pb "batchserve/gen/go/batchservev1"

	"google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/credentials/insecure"
)

// maxMsgBytes: grpc-go's default 4 MiB cap is too small for a batch of
// images (64 × 2 MiB at the extreme). Hitting it would surface as
// RESOURCE_EXHAUSTED, which looks exactly like queue backpressure but isn't.
// The worker raises its own limit to match.
const maxMsgBytes = 256 << 20

type Client struct {
	conn   *grpc.ClientConn
	client pb.InferenceWorkerClient
}

// Dial creates a client for addr. It does NOT wait for the worker to be up:
// grpc.NewClient connects lazily and keeps retrying in the background. That is
// what lets the gateway start before the worker; until the worker appears,
// calls fail fast with UNAVAILABLE (no WaitForReady), which the batcher passes
// on to clients, and once the worker is up they succeed with no gateway restart.
func Dial(addr string) (*Client, error) {
	conn, err := grpc.NewClient(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallSendMsgSize(maxMsgBytes),
			grpc.MaxCallRecvMsgSize(maxMsgBytes),
		),
		// The default reconnect backoff grows to 2 minutes. A worker that takes
		// a few seconds to load its model should be found again within about
		// a second of becoming ready, not minutes later.
		grpc.WithConnectParams(grpc.ConnectParams{
			Backoff: backoff.Config{
				BaseDelay:  100 * time.Millisecond,
				Multiplier: 1.6,
				Jitter:     0.2,
				MaxDelay:   time.Second,
			},
			MinConnectTimeout: time.Second,
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("dial worker %q: %w", addr, err)
	}
	return &Client{conn: conn, client: pb.NewInferenceWorkerClient(conn)}, nil
}

// InferBatch implements batcher.Inferer. The deadline comes from ctx; the
// batcher sets it to the fixed --worker-timeout.
func (c *Client) InferBatch(ctx context.Context, req *pb.InferBatchRequest) (*pb.InferBatchResponse, error) {
	return c.client.InferBatch(ctx, req)
}

func (c *Client) Close() error { return c.conn.Close() }
