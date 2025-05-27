const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

export interface RedactionStats {
  totalPIICount: number;
  byType: { [key: string]: number };
  byMethod: { [key: string]: number };
  byPage: { [key: string]: number };
}

export interface ProcessingResponse {
  job_id: string;
  originalFilename: string;
  status: string;
  metadata?: { [key: string]: string };
  redactionStats?: RedactionStats;
}

export interface AutoRedactionResponse {
  success: boolean;
  message?: string;
  redactedPdfUrl?: string;
  jobId?: string;
  redactionStats?: RedactionStats;
}

export const api = {
  async uploadForAutoRedaction(file: File): Promise<AutoRedactionResponse> {
    const formData = new FormData();
    formData.append('document', file);
    formData.append('options', JSON.stringify({
      "enable_redaction": true,
      "preserve_formatting": true,
    }));

    try {
      // Initial upload and processing request
      const response = await fetch(`${API_BASE_URL}/api/documents/process`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to process document');
      }

      const processingResponse: ProcessingResponse = await response.json();
      const jobId = processingResponse.job_id;

      if (!jobId) {
        throw new Error('No job ID received from server');
      }

      // Poll for job completion
      let attempts = 0;
      const maxAttempts = 60; // 5 minutes with 5-second intervals
      
      while (attempts < maxAttempts) {
        const statusResponse = await fetch(`${API_BASE_URL}/api/documents/status/${jobId}`);
        
        if (!statusResponse.ok) {
          if (statusResponse.status === 404) {
            throw new Error('Job not found');
          }
          throw new Error('Failed to check processing status');
        }

        const statusData = await statusResponse.json();
        
        if (statusData.status === 'completed') {
          // Get the redacted document
          return {
            success: true,
            redactedPdfUrl: `${API_BASE_URL}/api/documents/download/${jobId}`,
            jobId: jobId,
            redactionStats: statusData.redactionStats,
          };
        } else if (statusData.status === 'error' || statusData.status === 'failed') {
          const errorMessage = statusData.metadata?.error || 'Processing failed';
          throw new Error(errorMessage);
        }

        // Wait before next attempt
        await new Promise(resolve => setTimeout(resolve, 3000));
        attempts++;
      }

      throw new Error('Processing timeout - please try again');
    } catch (error) {
      console.error('Error during auto-redaction:', error);
      return {
        success: false,
        message: error instanceof Error ? error.message : 'An unknown error occurred',
      };
    }
  },

  async translateDocument(file: File, targetLanguage: 'en' | 'id'): Promise<AutoRedactionResponse> {
    const formData = new FormData();
    formData.append('document', file);
    formData.append('options', JSON.stringify({
      enableTranslation: true,
      targetLanguage: targetLanguage,
      preserveFormatting: true,
    }));

    try {
      const response = await fetch(`${API_BASE_URL}/api/documents/process`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to translate document');
      }

      const processingResponse: ProcessingResponse = await response.json();
      const jobId = processingResponse.job_id;

      if (!jobId) {
        throw new Error('No job ID received from server');
      }

      // Poll for job completion
      let attempts = 0;
      const maxAttempts = 60; // 5 minutes with 5-second intervals
      
      while (attempts < maxAttempts) {
        const statusResponse = await fetch(`${API_BASE_URL}/api/documents/status/${jobId}`);
        
        if (!statusResponse.ok) {
          if (statusResponse.status === 404) {
            throw new Error('Job not found');
          }
          throw new Error('Failed to check processing status');
        }

        const statusData = await statusResponse.json();
        
        if (statusData.status === 'completed') {
          return {
            success: true,
            redactedPdfUrl: `${API_BASE_URL}/api/documents/download/${jobId}`,
            jobId: jobId,
          };
        } else if (statusData.status === 'error' || statusData.status === 'failed') {
          const errorMessage = statusData.metadata?.error || 'Translation failed';
          throw new Error(errorMessage);
        }

        // Wait before next attempt
        await new Promise(resolve => setTimeout(resolve, 5000));
        attempts++;
      }

      throw new Error('Translation timeout - please try again');
    } catch (error) {
      console.error('Error during translation:', error);
      return {
        success: false,
        message: error instanceof Error ? error.message : 'An unknown error occurred',
      };
    }
  },
}; 