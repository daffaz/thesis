package handlers

import (
	"context"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"server_pdf_processor/internal/client"
	"server_pdf_processor/internal/models"
)

// DocumentHandler handles document processing requests
type DocumentHandler struct {
	pdfClient *client.PDFProcessorClient
}

// NewDocumentHandler creates a new document handler
func NewDocumentHandler(pdfClient *client.PDFProcessorClient) *DocumentHandler {
	return &DocumentHandler{
		pdfClient: pdfClient,
	}
}

// RegisterRoutes registers the document handler routes
func (h *DocumentHandler) RegisterRoutes(router *gin.Engine) {
	documents := router.Group("/api/documents")
	{
		documents.POST("/process", h.ProcessDocument)
		documents.GET("/status/:jobID", h.GetStatus)
		documents.GET("/download/:jobID", h.DownloadDocument)
		documents.POST("/detect-pii", h.DetectPII)
	}
}

// ProcessDocument handles document processing requests
func (h *DocumentHandler) ProcessDocument(c *gin.Context) {
	// Parse multipart form
	if err := c.Request.ParseMultipartForm(32 << 20); err != nil { // 32MB max
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:   "INVALID_REQUEST",
			Code:    http.StatusBadRequest,
			Message: "Invalid request: " + err.Error(),
		})
		return
	}

	// Get the uploaded file
	file, header, err := c.Request.FormFile("document")
	if err != nil {
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:   "MISSING_FILE",
			Code:    http.StatusBadRequest,
			Message: "No document file provided",
		})
		return
	}
	defer file.Close()

	// Validate file type
	filename := header.Filename
	if !isPDF(filename) {
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:   "INVALID_FILE_TYPE",
			Code:    http.StatusBadRequest,
			Message: "Only PDF files are supported",
		})
		return
	}

	// Parse processing options
	var options models.ProcessingOptions
	if optionsStr := c.PostForm("options"); optionsStr != "" {
		if err := c.ShouldBindJSON(&options); err != nil {
			c.JSON(http.StatusBadRequest, models.ErrorResponse{
				Error:   "INVALID_OPTIONS",
				Code:    http.StatusBadRequest,
				Message: "Invalid processing options: " + err.Error(),
			})
			return
		}
	} else {
		// Default options
		options = models.ProcessingOptions{
			EnableRedaction:    true,
			PreserveFormatting: true,
		}
	}

	// Convert to client options
	clientOptions := client.PDFProcessingOptions{
		EnableRedaction:    options.EnableRedaction,
		RedactionTypes:     options.RedactionTypes,
		EnableTranslation:  options.EnableTranslation,
		TargetLanguage:     options.TargetLanguage,
		PreserveFormatting: options.PreserveFormatting,
	}

	// Process the document
	ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
	defer cancel()

	result, err := h.pdfClient.ProcessDocument(ctx, file, filename, clientOptions)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.ErrorResponse{
			Error:   "PROCESSING_ERROR",
			Code:    http.StatusInternalServerError,
			Message: "Error processing document: " + err.Error(),
		})
		return
	}

	// Map the response
	response := models.ProcessingResponse{
		JobID:            result.JobID,
		OriginalFilename: result.OriginalFilename,
		Status:           "processing", // Initial status
		Metadata:         result.Metadata,
	}

	if result.RedactionStats != nil {
		response.RedactionStats = &models.RedactionStats{
			TotalPIICount: result.RedactionStats.TotalPIICount,
			ByType:        result.RedactionStats.ByType,
			ByMethod:      result.RedactionStats.ByMethod,
			ByPage:        result.RedactionStats.ByPage,
		}
	}

	c.JSON(http.StatusAccepted, response)
}

// GetStatus handles job status requests
func (h *DocumentHandler) GetStatus(c *gin.Context) {
	jobID := c.Param("jobID")
	if jobID == "" {
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:   "MISSING_JOB_ID",
			Code:    http.StatusBadRequest,
			Message: "Job ID is required",
		})
		return
	}

	// Check if the job ID is valid
	if _, err := uuid.Parse(jobID); err != nil {
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:   "INVALID_JOB_ID",
			Code:    http.StatusBadRequest,
			Message: "Invalid job ID format",
		})
		return
	}

	// Get the job status
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	status, err := h.pdfClient.GetStatus(ctx, jobID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.ErrorResponse{
			Error:   "STATUS_ERROR",
			Code:    http.StatusInternalServerError,
			Message: "Error getting job status: " + err.Error(),
		})
		return
	}

	// Map the response
	response := models.StatusResponse{
		JobID:      status.JobID,
		Status:     status.Status,
		OutputFile: status.OutputFile,
	}

	c.JSON(http.StatusOK, response)
}

// DownloadDocument handles document download requests
// Note: In a real implementation, this would retrieve the document from storage
func (h *DocumentHandler) DownloadDocument(c *gin.Context) {
	jobID := c.Param("jobID")
	if jobID == "" {
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:   "MISSING_JOB_ID",
			Code:    http.StatusBadRequest,
			Message: "Job ID is required",
		})
		return
	}

	// Get the job status to ensure it's completed
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	status, err := h.pdfClient.GetStatus(ctx, jobID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.ErrorResponse{
			Error:   "STATUS_ERROR",
			Code:    http.StatusInternalServerError,
			Message: "Error getting job status: " + err.Error(),
		})
		return
	}

	if status.Status != "completed" {
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:   "JOB_NOT_COMPLETED",
			Code:    http.StatusBadRequest,
			Message: "Document processing not yet completed",
		})
		return
	}

	// In a real implementation, you would retrieve the document from storage
	// and stream it back to the client.
	// For now, this is a placeholder that returns a 501 Not Implemented
	c.JSON(http.StatusNotImplemented, models.ErrorResponse{
		Error:   "NOT_IMPLEMENTED",
		Code:    http.StatusNotImplemented,
		Message: "Document download not yet implemented",
	})
}

// DetectPII handles PII detection requests (streaming)
func (h *DocumentHandler) DetectPII(c *gin.Context) {
	// Parse multipart form
	if err := c.Request.ParseMultipartForm(32 << 20); err != nil { // 32MB max
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:   "INVALID_REQUEST",
			Code:    http.StatusBadRequest,
			Message: "Invalid request: " + err.Error(),
		})
		return
	}

	// Get the uploaded file
	file, header, err := c.Request.FormFile("document")
	if err != nil {
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:   "MISSING_FILE",
			Code:    http.StatusBadRequest,
			Message: "No document file provided",
		})
		return
	}
	defer file.Close()

	// Validate file type
	filename := header.Filename
	if !isPDF(filename) {
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:   "INVALID_FILE_TYPE",
			Code:    http.StatusBadRequest,
			Message: "Only PDF files are supported",
		})
		return
	}

	// Generate a unique job ID for this operation
	jobID := uuid.New().String()

	// Start the PII detection stream
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Minute)
	defer cancel()

	resultChan, err := h.pdfClient.StreamPIIDetection(ctx, file, filename)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.ErrorResponse{
			Error:   "PII_DETECTION_ERROR",
			Code:    http.StatusInternalServerError,
			Message: "Error starting PII detection: " + err.Error(),
		})
		return
	}

	// Collect all results
	var results []models.PIIDetectionResult
	for result := range resultChan {
		results = append(results, models.PIIDetectionResult{
			PIIText: result.PIIText,
			PIIType: result.PIIType,
			Page:    result.Page,
			Start:   result.Start,
			End:     result.End,
		})
	}

	// Return the results
	response := models.PIIDetectionResponse{
		JobID:   jobID,
		Results: results,
	}

	c.JSON(http.StatusOK, response)
}

// isPDF checks if a filename has a PDF extension
func isPDF(filename string) bool {
	return len(filename) >= 4 && filename[len(filename)-4:] == ".pdf"
}
