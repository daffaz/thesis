package models

// ProcessingOptions represents the options for document processing
type ProcessingOptions struct {
	EnableRedaction    bool     `json:"enable_redaction"`
	RedactionTypes     []string `json:"redaction_types,omitempty"`
	EnableTranslation  bool     `json:"enable_translation"`
	TargetLanguage     string   `json:"target_language,omitempty"`
	PreserveFormatting bool     `json:"preserve_formatting"`
}

// ProcessingResponse represents the response from a document processing request
type ProcessingResponse struct {
	JobID            string            `json:"job_id"`
	OriginalFilename string            `json:"original_filename"`
	Status           string            `json:"status"`
	Metadata         map[string]string `json:"metadata,omitempty"`
	RedactionStats   *RedactionStats   `json:"redaction_stats,omitempty"`
}

// RedactionStats represents statistics about PII detection and redaction
type RedactionStats struct {
	TotalPIICount int32            `json:"total_pii_count"`
	ByType        map[string]int32 `json:"by_type,omitempty"`
	ByMethod      map[string]int32 `json:"by_method,omitempty"`
	ByPage        map[string]int32 `json:"by_page,omitempty"`
}

// StatusResponse represents the response from a job status request
type StatusResponse struct {
	JobID      string `json:"job_id"`
	Status     string `json:"status"`
	OutputFile string `json:"output_file,omitempty"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error   string `json:"error"`
	Code    int    `json:"code"`
	Message string `json:"message,omitempty"`
}

// PIIDetectionResult represents a detected PII item
type PIIDetectionResult struct {
	PIIText string `json:"pii_text"`
	PIIType string `json:"pii_type"`
	Page    int32  `json:"page"`
	Start   int32  `json:"start"`
	End     int32  `json:"end"`
}

// PIIDetectionResponse represents the response from a PII detection request
type PIIDetectionResponse struct {
	JobID   string               `json:"job_id"`
	Results []PIIDetectionResult `json:"results"`
}

// HealthResponse represents the response from a health check
type HealthResponse struct {
	Status    string            `json:"status"`
	Services  map[string]string `json:"services,omitempty"`
	Version   string            `json:"version"`
	Timestamp string            `json:"timestamp"`
}
