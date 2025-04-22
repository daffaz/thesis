# Privacy-First Document Processor

A privacy-focused document processing system with PII redaction and multi-language translation capabilities.

## Architecture

This project uses a microservices architecture with the following components:

- **API Server** (Go): Main entry point for client applications, handles HTTP requests
- **PDF Processor** (Python): Core service for PDF processing, PII detection, and redaction
- **Storage** (MinIO): Local S3-compatible storage for documents (planned)
- **Translation Service**: Neural machine translation (planned)

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/document-processor.git
   cd document-processor
   ```

2. Create necessary directories:
   ```bash
   mkdir -p data/input data/output data/temp
   ```

3. Start the services:
   ```bash
   docker-compose up --build
   ```

4. The API will be available at http://localhost:8080

## API Endpoints

### Document Processing

- **POST /api/documents/process**
  - Process a document with optional redaction and/or translation
  - Form parameters:
    - `document`: PDF file
    - `options`: JSON string with processing options

- **GET /api/documents/status/:jobID**
  - Check the status of a processing job

- **GET /api/documents/download/:jobID**
  - Download a processed document

- **POST /api/documents/detect-pii**
  - Detect PII in a document without processing it
  - Form parameters:
    - `document`: PDF file

### Health Check

- **GET /health** or **GET /api/health**
  - Check the health of the services

## Development

### Project Structure

```
.
├── api-server/              # Go API server
│   ├── cmd/
│   ├── internal/
│   └── pkg/
├── pdf-processor/           # Python PDF processor
│   ├── protos/              # Protocol buffer definitions
│   └── src/                 # Python source code
├── data/                    # Shared data directory
│   ├── input/               # Input documents
│   ├── output/              # Processed documents
│   └── temp/                # Temporary files
└── docker-compose.yml       # Docker Compose configuration
```

### Building the Services

Each service can be built individually:

```bash
# Build the API server
docker-compose build api-server

# Build the PDF processor
docker-compose build pdf-processor
```

### Running Tests

```bash
# Run tests for the API server
cd api-server
go test ./...

# Run tests for the PDF processor
cd pdf-processor
python -m pytest
```

## Roadmap

- [ ] Implement document storage with MinIO
- [ ] Add translation service with NLLB model
- [ ] Implement user authentication
- [ ] Add support for more document types (DOCX, etc.)
- [ ] Enhance PII detection with custom rules

## License

This project is licensed under the MIT License - see the LICENSE file for details.