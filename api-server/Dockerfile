FROM golang:1.24-alpine AS builder

# Install necessary tools
RUN apk add --no-cache git protoc protobuf-dev dos2unix

# Set up working directory
WORKDIR /app

# Install Go gRPC tools
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.28 && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.2

# Copy go.mod and go.sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy the source code
COPY . .

# Generate protobuf code
RUN mkdir -p pkg/pdf/pdf_processor && \
    dos2unix script/generate_proto.sh && \
    chmod +x script/generate_proto.sh && \
    ./script/generate_proto.sh

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -gcflags="all=-N -l" -a -installsuffix cgo -o api-server ./cmd/server

# Create a minimal runtime container
FROM alpine:3.18

# Add CA certificates for HTTPS
RUN apk --no-cache add ca-certificates

WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/api-server .

# Copy config files
COPY --from=builder /app/internal/config ./config

# Expose port
EXPOSE 8080

# Set the entry point
ENTRYPOINT ["/app/api-server"]