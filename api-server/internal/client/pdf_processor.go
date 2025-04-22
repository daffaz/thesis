package client

import (
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"time"

	"server_pdf_processor/internal/models"
	"server_pdf_processor/pkg/pdf/pdf_processor/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// PDFProcessorClient wraps the gRPC client for the PDF processor service
type PDFProcessorClient struct {
	client pdf_processor.PDFProcessorClient
	conn   *grpc.ClientConn
}

// PDFProcessingOptions contains options for PDF processing
type PDFProcessingOptions struct {
	EnableRedaction    bool
	RedactionTypes     []string
	EnableTranslation  bool
	TargetLanguage     string
	SourceLanguage     string
	PreserveFormatting bool
}

// ProcessingResult contains the result of processing a document
type ProcessingResult struct {
	JobID            string
	OriginalFilename string
	Metadata         map[string]string
	RedactionStats   *RedactionStats
}

// RedactionStats contains statistics about PII detection and redaction
type RedactionStats struct {
	TotalPIICount int32
	ByType        map[string]int32
	ByMethod      map[string]int32
	ByPage        map[string]int32
}

// PIIDetectionResult represents a detected PII item
type PIIDetectionResult struct {
	PIIText string
	PIIType string
	Page    int32
	Start   int32
	End     int32
}

// JobStatus represents the status of a processing job
type JobStatus struct {
	JobID      string
	Status     string
	OutputFile string
}

// NewPDFProcessorClient creates a new client for the PDF processor service
func NewPDFProcessorClient(serverAddr string) (*PDFProcessorClient, error) {
	// Set up a connection to the server
	conn, err := grpc.Dial(serverAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to PDF processor: %v", err)
	}

	client := pdf_processor.NewPDFProcessorClient(conn)
	return &PDFProcessorClient{
		client: client,
		conn:   conn,
	}, nil
}

// Close closes the client connection
func (c *PDFProcessorClient) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// ProcessDocument sends a document for processing
func (c *PDFProcessorClient) ProcessDocument(ctx context.Context, file io.Reader, filename string, options PDFProcessingOptions) (*models.ProcessingResponse, error) {
	// Read the file content
	fileContent, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	// Create the gRPC request
	req := &pdf_processor.ProcessRequest{
		Document: fileContent,
		Filename: filename,
		Options: &pdf_processor.ProcessingOptions{
			EnableRedaction:    options.EnableRedaction,
			RedactionTypes:     options.RedactionTypes,
			EnableTranslation:  options.EnableTranslation,
			SourceLanguage:     options.SourceLanguage,
			TargetLanguage:     options.TargetLanguage,
			PreserveFormatting: options.PreserveFormatting,
		},
	}

	// Call the gRPC service
	resp, err := c.client.ProcessDocument(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("gRPC call failed: %w", err)
	}

	// Map the response to our model
	result := &models.ProcessingResponse{
		JobID:            resp.JobId,
		OriginalFilename: resp.OriginalFilename,
		Status:           "processing",
		Metadata:         make(map[string]string),
	}

	// Copy metadata
	for k, v := range resp.Metadata {
		result.Metadata[k] = v
	}

	// Map redaction stats if available
	if resp.RedactionStats != nil {
		result.RedactionStats = &models.RedactionStats{
			TotalPIICount: resp.RedactionStats.TotalPiiCount,
			ByType:        make(map[string]int64),
			ByMethod:      make(map[string]int64),
			ByPage:        make(map[string]int64),
		}

		for k, v := range resp.RedactionStats.ByType {
			result.RedactionStats.ByType[k] = v
		}

		for k, v := range resp.RedactionStats.ByMethod {
			result.RedactionStats.ByMethod[k] = v
		}

		for k, v := range resp.RedactionStats.ByPage {
			result.RedactionStats.ByPage[k] = v
		}
	}

	return result, nil
}

// GetStatus checks the status of a processing job
func (c *PDFProcessorClient) GetStatus(ctx context.Context, jobID string) (*JobStatus, error) {
	// Prepare the request
	req := &pdf_processor.StatusRequest{
		JobId: jobID,
	}

	// Set a timeout for the request
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	// Send the request to the server
	resp, err := c.client.GetStatus(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("error getting job status: %v", err)
	}

	// Map the response to our internal model
	status := &JobStatus{
		JobID:      resp.JobId,
		Status:     resp.Status,
		OutputFile: resp.OutputFile,
	}

	return status, nil
}

// StreamPIIDetection streams PII detection results for a document
func (c *PDFProcessorClient) StreamPIIDetection(
	ctx context.Context,
	documentReader io.Reader,
	filename string,
) (chan *PIIDetectionResult, error) {
	// Read the document data
	documentData, err := ioutil.ReadAll(documentReader)
	if err != nil {
		return nil, fmt.Errorf("failed to read document: %v", err)
	}

	// Prepare the request
	req := &pdf_processor.DocumentStream{
		Document: documentData,
		Filename: filename,
	}

	// Create a result channel
	resultChan := make(chan *PIIDetectionResult, 100) // Buffer size of 100

	// Set a timeout for the stream
	ctx, cancel := context.WithTimeout(ctx, 5*time.Minute)

	// Start a goroutine to process the stream
	go func() {
		defer cancel()
		defer close(resultChan)

		// Start the stream
		stream, err := c.client.StreamPIIDetection(ctx, req)
		if err != nil {
			log.Printf("Error starting PII detection stream: %v", err)
			return
		}

		// Process results from the stream
		for {
			resp, err := stream.Recv()
			if err == io.EOF {
				// End of stream
				break
			}
			if err != nil {
				log.Printf("Error receiving PII detection result: %v", err)
				break
			}

			// Map the response to our internal model
			result := &PIIDetectionResult{
				PIIText: resp.PiiText,
				PIIType: resp.PiiType,
				Page:    resp.Page,
				Start:   resp.Start,
				End:     resp.End,
			}

			// Send the result to the channel
			select {
			case resultChan <- result:
				// Result sent successfully
			case <-ctx.Done():
				// Context cancelled or timed out
				return
			}
		}
	}()

	return resultChan, nil
}
