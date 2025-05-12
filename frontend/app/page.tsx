'use client';

import { useState } from 'react';
import DocumentUploader from '@/components/DocumentUploader';
import PDFViewer from '@/components/PDFViewer';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [processingMode, setProcessingMode] = useState<'auto' | 'manual' | 'combined'>('auto');

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
  };

  return (
    <main className="min-h-screen bg-gray-50">
      <div className="max-w-5xl mx-auto px-4 py-12">
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
          <div className="p-8">
            <h1 className="text-2xl font-semibold text-gray-900 mb-8">
              <a href="/">
                UAI PDF Processor
              </a>
            </h1>
            
            {!selectedFile ? (
              <DocumentUploader 
                onFileSelect={handleFileSelect}
                onModeSelect={setProcessingMode}
                selectedMode={processingMode}
              />
            ) : (
              <PDFViewer 
                file={selectedFile}
                mode={processingMode}
                onReset={() => setSelectedFile(null)}
              />
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
