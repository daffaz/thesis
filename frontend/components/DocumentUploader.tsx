import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { DocumentArrowUpIcon } from '@heroicons/react/24/outline';

interface DocumentUploaderProps {
  onFileSelect: (file: File) => void;
  onModeSelect: (mode: 'auto' | 'manual' | 'combined') => void;
  selectedMode: 'auto' | 'manual' | 'combined';
}

export default function DocumentUploader({
  onFileSelect,
  onModeSelect,
  selectedMode,
}: DocumentUploaderProps) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    maxFiles: 1,
  });

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-lg font-medium text-gray-700 mb-4">Processing Mode</h2>
        <div className="inline-flex rounded-lg p-1 bg-gray-100">
          {(['auto', 'manual', 'combined'] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => onModeSelect(mode)}
              className={`
                px-6 py-2 rounded-lg text-sm font-medium transition-all
                ${selectedMode === mode
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
                }
              `}
            >
              {mode === 'auto' && 'Auto Redaction'}
              {mode === 'manual' && 'Manual Redaction'}
              {mode === 'combined' && 'Combined Redaction'}
            </button>
          ))}
        </div>
      </div>

      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-xl p-12
          flex flex-col items-center justify-center
          cursor-pointer transition-all
          ${isDragActive
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-200 hover:border-gray-300'
          }
        `}
      >
        <input {...getInputProps()} />
        <div className="p-4 rounded-full bg-gray-100 mb-4">
          <DocumentArrowUpIcon className="h-8 w-8 text-gray-400" />
        </div>
        <p className="text-lg font-medium text-gray-700">
          {isDragActive ? 'Drop your PDF here' : 'Drag & drop your PDF here'}
        </p>
        <p className="mt-2 text-sm text-gray-500">
          or click to select a file
        </p>
      </div>
    </div>
  );
} 