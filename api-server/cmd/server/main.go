package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"server_pdf_processor/internal/client"
	"server_pdf_processor/internal/config"
	"server_pdf_processor/internal/handlers"
	"server_pdf_processor/internal/middleware"

	"github.com/gin-gonic/gin"
)

// Version information
var (
	version = "0.1.0" // Replace with build-time value in real implementation
)

func main() {
	// Load configuration
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Set up logging
	setupLogging(cfg.Logging.Level)

	// Set up PDF processor client
	pdfProcessorAddr := fmt.Sprintf("%s:%d",
		cfg.Services.PDFProcessor.Host,
		cfg.Services.PDFProcessor.Port)

	pdfClient, err := client.NewPDFProcessorClient(pdfProcessorAddr)
	if err != nil {
		log.Fatalf("Failed to create PDF processor client: %v", err)
	}
	defer pdfClient.Close()

	// Set up Gin router
	router := setupRouter(pdfClient)

	// Create HTTP server
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Server.Port),
		Handler:      router,
		ReadTimeout:  time.Duration(cfg.Server.ReadTimeout) * time.Second,
		WriteTimeout: time.Duration(cfg.Server.WriteTimeout) * time.Second,
	}

	// Run server in a goroutine
	go func() {
		log.Printf("Server listening on port %d", cfg.Server.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Wait for interrupt signal to gracefully shut down the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down server...")

	// Create a deadline for the shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exiting")
}

// setupRouter configures the Gin router with routes and middleware
func setupRouter(pdfClient *client.PDFProcessorClient) *gin.Engine {
	// Create router
	router := gin.New()

	// Add middleware
	router.Use(gin.Recovery())
	router.Use(middleware.RequestLoggerMiddleware())
	router.Use(middleware.CORSMiddleware())
	router.Use(middleware.ErrorMiddleware())
	router.Use(middleware.MaxBodySizeMiddleware(32 << 20)) // 32MB max

	// Create handlers
	healthHandler := handlers.NewHealthHandler(pdfClient, version)
	documentHandler := handlers.NewDocumentHandler(pdfClient)

	// Register routes
	healthHandler.RegisterRoutes(router)
	documentHandler.RegisterRoutes(router)

	return router
}

// setupLogging configures the logging system
func setupLogging(level string) {
	// In a real application, you might use a more sophisticated
	// logging library like zap or logrus
	log.SetOutput(os.Stdout)
	log.SetFlags(log.LstdFlags | log.Lshortfile)
}
