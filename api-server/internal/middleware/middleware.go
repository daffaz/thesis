package middleware

import (
	"bytes"
	"io"
	"log"
	"time"

	"github.com/gin-gonic/gin"
)

// RequestLoggerMiddleware logs information about each HTTP request
func RequestLoggerMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Start timer
		start := time.Now()
		path := c.Request.URL.Path
		raw := c.Request.URL.RawQuery

		// Process request
		c.Next()

		// Calculate latency
		latency := time.Since(start)

		// Get status code and client IP
		statusCode := c.Writer.Status()
		clientIP := c.ClientIP()
		method := c.Request.Method

		// Log the request details
		log.Printf("[API] %s | %3d | %13v | %15s | %-7s %s%s",
			method,
			statusCode,
			latency,
			clientIP,
			method,
			path,
			func() string {
				if raw != "" {
					return "?" + raw
				}
				return ""
			}(),
		)
	}
}

// CORSMiddleware handles Cross-Origin Resource Sharing
func CORSMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Credentials", "true")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, accept, origin, Cache-Control, X-Requested-With")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS, GET, PUT, DELETE")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	}
}

// ErrorMiddleware handles errors from the handlers
func ErrorMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Next()

		// Only handle errors if the response hasn't been written yet
		if !c.Writer.Written() {
			if len(c.Errors) > 0 {
				// Get the last error
				err := c.Errors.Last()

				// Log the error
				log.Printf("[ERROR] %s", err.Error())

				// Return a JSON error response
				c.JSON(c.Writer.Status(), gin.H{
					"error":   "INTERNAL_SERVER_ERROR",
					"code":    c.Writer.Status(),
					"message": err.Error(),
				})
			}
		}
	}
}

// MaxBodySizeMiddleware limits the size of request bodies
func MaxBodySizeMiddleware(maxSize int64) gin.HandlerFunc {
	return func(c *gin.Context) {
		var buff bytes.Buffer

		// Read at most maxSize bytes
		readBytes, _ := io.CopyN(&buff, c.Request.Body, maxSize+1)

		if readBytes > maxSize {
			c.AbortWithStatusJSON(413, gin.H{
				"error":   "REQUEST_ENTITY_TOO_LARGE",
				"code":    413,
				"message": "Request body too large",
			})
			return
		}

		// Replace the request body with our buffered copy
		c.Request.Body = io.NopCloser(&buff)
		c.Next()
	}
}
