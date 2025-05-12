'use client';

import { useEffect, useRef, useState } from 'react';
import { PDFDocument, rgb } from 'pdf-lib';
import type { PDFDocumentProxy, PageViewport, RenderTask } from 'pdfjs-dist';

// We'll initialize pdfjsLib dynamically
let pdfjsLib: any = null;

interface PDFViewerProps {
  file: File;
  mode: 'auto' | 'manual' | 'combined';
  onReset: () => void;
}

interface RedactionArea {
  x: number;
  y: number;
  width: number;
  height: number;
  pageIndex: number;
}

export default function PDFViewer({ file, mode, onReset }: PDFViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const currentRenderTask = useRef<RenderTask | null>(null);
  const [pdfDoc, setPdfDoc] = useState<PDFDocumentProxy | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [scale, setScale] = useState(1.5);
  const [redactionAreas, setRedactionAreas] = useState<RedactionArea[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const renderPage = async (doc: PDFDocumentProxy, pageNumber: number) => {
    if (!doc) return;

    try {
      // Get the canvas element
      const canvas = canvasRef.current;
      if (!canvas) {
        throw new Error('Canvas element not found');
      }

      // Cancel any ongoing render task
      if (currentRenderTask.current) {
        await currentRenderTask.current.cancel();
        currentRenderTask.current = null;
      }

      // Get the page
      const page = await doc.getPage(pageNumber);
      const viewport = page.getViewport({ scale });

      // Set canvas dimensions
      canvas.height = viewport.height;
      canvas.width = viewport.width;

      // Get the rendering context
      const context = canvas.getContext('2d');
      if (!context) {
        throw new Error('Could not get canvas context');
      }

      // Clear the canvas
      context.clearRect(0, 0, canvas.width, canvas.height);

      try {
        // Start new render task
        currentRenderTask.current = page.render({
          canvasContext: context,
          viewport,
        });

        // Wait for render to complete
        await currentRenderTask.current.promise;
        currentRenderTask.current = null;

        // Draw redaction areas after render is complete
        context.fillStyle = '#000000';
        redactionAreas
          .filter(area => area.pageIndex === pageNumber - 1)
          .forEach(area => {
            context.fillRect(area.x, area.y, area.width, area.height);
          });
      } catch (error: any) {
        if (error?.message !== 'Rendering cancelled, page 1') {
          console.error('Error during page render:', error);
          throw error;
        }
      }
    } catch (error) {
      console.error('Error in renderPage:', error);
      throw error;
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (mode === 'auto' || isLoading) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setIsDrawing(true);
    setStartPoint({ x, y });
  };

  const handleMouseMove = async (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !startPoint || !canvasRef.current || mode === 'auto' || isLoading) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const context = canvas.getContext('');
    if (!context || !pdfDoc) return;

    // Redraw the page
    await renderPage(pdfDoc, currentPage);

    // Draw the current selection rectangle
    context.fillStyle = 'rgba(0, 0, 0, 0.3)';
    context.fillRect(
      startPoint.x,
      startPoint.y,
      x - startPoint.x,
      y - startPoint.y
    );
  };

  const handleMouseUp = async (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !startPoint || !canvasRef.current || mode === 'auto' || isLoading) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Add new redaction area
    const newArea: RedactionArea = {
      x: Math.min(startPoint.x, x),
      y: Math.min(startPoint.y, y),
      width: Math.abs(x - startPoint.x),
      height: Math.abs(y - startPoint.y),
      pageIndex: currentPage - 1,
    };

    // Update redaction areas state and immediately draw the new area
    setRedactionAreas(prev => {
      const updatedAreas = [...prev, newArea];
      
      // Immediately draw the new redaction area
      const context = canvas.getContext('2d');
      if (context) {
        context.fillStyle = '#000000';
        context.fillRect(newArea.x, newArea.y, newArea.width, newArea.height);
      }
      
      return updatedAreas;
    });

    setIsDrawing(false);
    setStartPoint(null);
  };

  const applyRedactions = async () => {
    if (!pdfDoc) return;

    try {
      // Create a new PDF document using pdf-lib
      const pdfBytes = await file.arrayBuffer();
      const pdfLibDoc = await PDFDocument.load(pdfBytes);

      // Apply redactions to each page
      for (const area of redactionAreas) {
        const page = pdfLibDoc.getPages()[area.pageIndex];
        const pageHeight = page.getHeight();
        const pageWidth = page.getWidth();
        
        // Get the canvas for coordinate scaling
        const canvas = canvasRef.current;
        if (!canvas) continue;
        
        // Calculate scale factors
        const scaleX = pageWidth / canvas.width;
        const scaleY = pageHeight / canvas.height;
        
        // Transform coordinates
        const pdfX = area.x * scaleX;
        const pdfY = pageHeight - ((area.y + area.height) * scaleY); // Flip Y coordinate
        const pdfWidth = area.width * scaleX;
        const pdfHeight = area.height * scaleY;

        page.drawRectangle({
          x: pdfX,
          y: pdfY,
          width: pdfWidth,
          height: pdfHeight,
          color: rgb(0, 0, 0),
        });
      }

      // Save the redacted PDF
      const redactedPdfBytes = await pdfLibDoc.save();
      const blob = new Blob([redactedPdfBytes], { type: 'application/pdf' });
      const url = URL.createObjectURL(blob);

      // Download the redacted PDF
      const a = document.createElement('a');
      a.href = url;
      a.download = `redacted-${file.name}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error applying redactions:', error);
    }
  };

  // Initialize PDF.js
  useEffect(() => {
    let mounted = true;
    
    async function initPdfJs() {
      if (typeof window === 'undefined') return;
      
      try {
        setError(null);
        setIsLoading(true);

        // Only initialize pdfjsLib once
        if (!pdfjsLib) {
          const pdfjs = await import('pdfjs-dist');
          const pdfjsWorker = await import('pdfjs-dist/build/pdf.worker.entry');
          pdfjs.GlobalWorkerOptions.workerSrc = pdfjsWorker.default;
          pdfjsLib = pdfjs;
        }
        
        const fileArrayBuffer = await file.arrayBuffer();
        const loadingTask = pdfjsLib.getDocument({ data: fileArrayBuffer });
        
        const loadedPdf = await loadingTask.promise;
        
        if (mounted) {
          setPdfDoc(loadedPdf);
          setTotalPages(loadedPdf.numPages);
          setCurrentPage(1);
        }
      } catch (error) {
        console.error('Error initializing PDF.js:', error);
        if (mounted) {
          setError('Failed to load PDF. Please try again.');
        }
      } finally {
        if (mounted) {
          setIsLoading(false);
        }
      }
    }

    initPdfJs();

    return () => {
      mounted = false;
      if (currentRenderTask.current) {
        currentRenderTask.current.cancel();
      }
    };
  }, [file]);

  // Handle page rendering
  useEffect(() => {
    if (!pdfDoc || !canvasRef.current) return;
    
    let mounted = true;

    async function render() {
      try {
        setIsLoading(true);
        if (pdfDoc) {
          await renderPage(pdfDoc, currentPage);
        }
      } catch (error) {
        console.error('Error rendering page:', error);
        if (mounted) {
          setError('Failed to render page. Please try again.');
        }
      } finally {
        if (mounted) {
          setIsLoading(false);
        }
      }
    }

    render();

    return () => {
      mounted = false;
      if (currentRenderTask.current) {
        currentRenderTask.current.cancel();
      }
    };
  }, [pdfDoc, currentPage]);

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center p-8 space-y-4">
        <div className="text-red-600">{error}</div>
        <button
          onClick={onReset}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Try Again
        </button>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      <div className="p-4 border-b border-gray-100">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <button
              onClick={() => {
                if (currentPage > 1 && !isLoading && pdfDoc) {
                  setCurrentPage(currentPage - 1);
                }
              }}
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={currentPage <= 1 || isLoading}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <span className="text-sm font-medium text-gray-700">
              Page {currentPage} of {totalPages}
            </span>
            <button
              onClick={() => {
                if (currentPage < totalPages && !isLoading && pdfDoc) {
                  setCurrentPage(currentPage + 1);
                }
              }}
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={currentPage >= totalPages || isLoading}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={onReset}
              className="px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={isLoading}
            >
              Cancel
            </button>
            <button
              onClick={applyRedactions}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
              disabled={isLoading || redactionAreas.length === 0}
            >
              <span>Apply Redactions</span>
              {isLoading && (
                <svg className="animate-spin ml-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="relative bg-gray-50">
        {isLoading && (
          <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center z-10">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-600 border-t-transparent"></div>
          </div>
        )}
        
        <div className="p-6 flex justify-center">
          <canvas
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            className="max-w-full rounded shadow-sm"
          />
        </div>
      </div>
    </div>
  );
} 