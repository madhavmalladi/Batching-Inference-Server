.PHONY: build test clean

build:
	go build -o bin/gateway ./cmd/gateway
	go build -o bin/fakeworker ./cmd/fakeworker

test:
	go vet ./...
	go test -race ./...

clean:
	rm -rf bin
