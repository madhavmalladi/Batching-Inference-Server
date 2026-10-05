# Root Makefile: thin wrapper. Each component owns its own Makefile
# (targets: build, test, clean). Only step 0 and post-merge sessions edit this file.

COMPONENTS := gateway worker loadgen deploy
PROTOC ?= protoc
BIN := $(CURDIR)/bin
PROTOC_GEN_GO_VERSION := v1.36.12
PROTOC_GEN_GO_GRPC_VERSION := v1.6.2

.PHONY: proto test build clean tools

tools:
	GOBIN=$(BIN) go install google.golang.org/protobuf/cmd/protoc-gen-go@$(PROTOC_GEN_GO_VERSION)
	GOBIN=$(BIN) go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@$(PROTOC_GEN_GO_GRPC_VERSION)

# Regenerates the FROZEN Go stubs in gen/go. Run only for an approved contract change.
proto: tools
	rm -rf gen/go/batchservev1
	PATH=$(BIN):$$PATH $(PROTOC) -I proto \
		--go_out=gen/go --go_opt=module=batchserve/gen/go \
		--go-grpc_out=gen/go --go-grpc_opt=module=batchserve/gen/go \
		proto/batchserve/v1/*.proto
	cd gen/go && go mod tidy && go build ./...

# Runs each component's tests, skipping components that don't exist yet.
test build clean:
	@for c in $(COMPONENTS); do \
		if [ -f $$c/Makefile ]; then echo "==> $$c: $@"; $(MAKE) -C $$c $@ || exit 1; \
		else echo "==> $$c: (not present, skipped)"; fi; \
	done
