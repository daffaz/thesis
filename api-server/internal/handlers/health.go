package handlers

import (
	"context"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"server_pdf_processor/internal/client"
	"server_pdf_processor/internal/models"
)

// HealthHandler handles health check requests
type HealthHandler struct {
	pdfClient *client.PDFProcessorClient
	version   string
}

// NewHealthHandler creates a new health handler
func NewHealthHandler(pdfClient *client.PDFProcessorClient, version string) *HealthHandler {
	return &HealthHandler{
		pdfClient: pdfClient,
		version:   version,
	}
}

// RegisterRoutes registers the health handler routes
func (h *HealthHandler) RegisterRoutes(router *gin.Engine) {
	router.GET("/health", h.HealthCheck)
	router.GET("/api/health", h.HealthCheck)
}

// HealthCheck handles health check requests
func (h *HealthHandler) HealthCheck(c *gin.Context) {
	serviceStatus := make(map[string]string)

	// Check PDF Processor service
	ctx, cancel := context.WithTimeout(c.Request.Context(), 2*time.Second)
	defer cancel()

	// We'll try to get a simple status from the PDF Processor
	// If it fails, we'll mark the service as down
	_, err := h.pdfClient.GetStatus(ctx, "health-check")
	if err != nil {
		serviceStatus["pdf-processor"] = "down"
	} else {
		serviceStatus["pdf-processor"] = "up"
	}

	// Overall status is "ok" if all services are up
	overallStatus := "ok"
	for _, status := range serviceStatus {
		if status != "up" {
			overallStatus = "degraded"
			break
		}
	}

	// Return the health check response
	response := models.HealthResponse{
		Status:    overallStatus,
		Services:  serviceStatus,
		Version:   h.version,
		Timestamp: time.Now().Format(time.RFC3339),
	}

	c.JSON(http.StatusOK, response)
}
