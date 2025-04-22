#!/bin/bash
# tools/generate_protos.sh

# Make sure protoc is installed
if ! command -v protoc &> /dev/null; then
    echo "protoc is not installed. Please install Protocol Buffers compiler."
    exit 1
fi

# Make sure necessary Go plugins are installed
if ! command -v protoc-gen-go &> /dev/null; then
    echo "Installing protoc-gen-go..."
    go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.28
fi

if ! command -v protoc-gen-go-grpc &> /dev/null; then
    echo "Installing protoc-gen-go-grpc..."
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.2
fi

# Create output directory if it doesn't exist
mkdir -p pkg/pdf/pdf_processor

# Generate Go code from proto definition
protoc --go_out=pkg/pdf/pdf_processor \
       --go_opt=paths=source_relative \
       --go-grpc_out=pkg/pdf/pdf_processor \
       --go-grpc_opt=paths=source_relative \
       proto/processor.proto

echo "Proto compilation complete!"
