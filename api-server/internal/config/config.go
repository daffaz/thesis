package config

import (
	"fmt"
	"strings"

	"github.com/spf13/viper"
)

// Config holds all configuration for the server
type Config struct {
	Server   ServerConfig
	Services ServicesConfig
	Logging  LoggingConfig
}

// ServerConfig holds the configuration for the HTTP server
type ServerConfig struct {
	Port         int
	ReadTimeout  int
	WriteTimeout int
}

// ServicesConfig holds the configuration for external services
type ServicesConfig struct {
	PDFProcessor PDFProcessorConfig
}

// PDFProcessorConfig holds the configuration for the PDF processor service
type PDFProcessorConfig struct {
	Host string
	Port int
}

// LoggingConfig holds the configuration for logging
type LoggingConfig struct {
	Level string
}

// LoadConfig loads configuration from environment variables and config files
func LoadConfig() (*Config, error) {
	// Set up viper
	v := viper.New()

	// Set default values
	setDefaults(v)

	// Setup environment variables
	v.SetEnvPrefix("APP")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	// Read in config file if it exists
	v.SetConfigName("config")
	v.SetConfigType("yaml")
	v.AddConfigPath(".")
	v.AddConfigPath("./config")

	if err := v.ReadInConfig(); err != nil {
		// It's okay if there's no config file
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("error reading config file: %w", err)
		}
	}

	// Parse the config
	var config Config
	if err := v.Unmarshal(&config); err != nil {
		return nil, fmt.Errorf("unable to decode config: %w", err)
	}

	return &config, nil
}

// setDefaults sets default configuration values
func setDefaults(v *viper.Viper) {
	// Server defaults
	v.SetDefault("server.port", 8080)
	v.SetDefault("server.readTimeout", 5)   // seconds
	v.SetDefault("server.writeTimeout", 10) // seconds

	// PDF Processor defaults
	v.SetDefault("services.pdfProcessor.host", "pdf-processor")
	v.SetDefault("services.pdfProcessor.port", 50051)

	// Logging defaults
	v.SetDefault("logging.level", "info")
}
