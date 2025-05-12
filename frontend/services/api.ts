const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

export interface AutoRedactionResponse {
  success: boolean;
  message?: string;
  redactedPdfUrl?: string;
}

export const api = {
  async uploadForAutoRedaction(file: File): Promise<AutoRedactionResponse> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/redact`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process document');
      }

      return await response.json();
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
    formData.append('file', file);
    formData.append('targetLanguage', targetLanguage);

    try {
      const response = await fetch(`${API_BASE_URL}/api/translate`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to translate document');
      }

      return await response.json();
    } catch (error) {
      console.error('Error during translation:', error);
      return {
        success: false,
        message: error instanceof Error ? error.message : 'An unknown error occurred',
      };
    }
  },
}; 