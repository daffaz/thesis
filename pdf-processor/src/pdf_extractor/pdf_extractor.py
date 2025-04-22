import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure, LTTextBox

class PDFExtractor:
    """
    Extracts text and structure from PDF documents using multiple methods
    to ensure high-quality extraction, including handling of scanned documents.
    """
    
    def __init__(self, enable_ocr: bool = True):
        """
        Initialize the PDF extractor.
        
        Args:
            enable_ocr: Whether to use OCR for scanned documents
        """
        self.enable_ocr = enable_ocr
        
    def extract_text(self, pdf_path: str) -> Dict[int, Dict]:
        """
        Extract text from a PDF file while preserving structure.
        Returns a dictionary with page numbers as keys and page content as values.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with page numbers as keys and structured content as values
        """
        # Validate file exists and is a PDF
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        # First try extraction with PyPDF2
        content = self._extract_with_pypdf2(pdf_path)
        
        # If content is empty or seems to be a scanned document, try pdfminer for more structure
        if self._is_low_text_content(content):
            content = self._extract_with_pdfminer(pdf_path)
            
        # If still low content and OCR is enabled, try OCR
        if self._is_low_text_content(content) and self.enable_ocr:
            content = self._extract_with_ocr(pdf_path)
        
        return content
    
    def _is_low_text_content(self, content: Dict[int, Dict]) -> bool:
        """
        Check if the extracted content seems to have low text content,
        which might indicate a scanned document.
        
        Args:
            content: Dictionary with page numbers as keys and page content
            
        Returns:
            True if content seems low, False otherwise
        """
        total_chars = sum(len(str(page_content.get('text', ''))) 
                          for page_content in content.values())
        total_pages = len(content)
        
        # Heuristic: average of less than 100 characters per page indicates low content
        return total_chars / max(total_pages, 1) < 100
    
    def _extract_with_pypdf2(self, pdf_path: str) -> Dict[int, Dict]:
        """
        Extract text using PyPDF2, which is fast but basic.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with page numbers as keys and content as values
        """
        result = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    
                    # Store page content with metadata
                    result[i+1] = {
                        'text': page_text,
                        'metadata': {
                            'page_number': i+1,
                            'total_pages': len(reader.pages),
                            'has_images': len(page.images) > 0
                        }
                    }
        except Exception as e:
            print(f"Error extracting with PyPDF2: {e}")
            # Return empty dict on error
            return {1: {'text': '', 'metadata': {'error': str(e)}}}
        
        return result
    
    def _extract_with_pdfminer(self, pdf_path: str) -> Dict[int, Dict]:
        """
        Extract text using PDFMiner which preserves more structure.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with page numbers as keys and structured content
        """
        result = {}
        
        try:
            # Extract pages with pdfminer
            for i, page_layout in enumerate(extract_pages(pdf_path)):
                page_number = i + 1
                paragraphs = []
                tables = []
                
                # Process each element on the page
                for element in page_layout:
                    if isinstance(element, LTTextBox):
                        paragraphs.append(element.get_text())
                    
                    # Simple table detection based on grouped rectangles
                    # This is a simplification - real table detection is more complex
                    if isinstance(element, LTRect) and hasattr(element, 'width') and hasattr(element, 'height'):
                        if element.width > 50 and element.height > 10:  # Potential table cell
                            # Look for nearby text
                            for text_elem in page_layout:
                                if isinstance(text_elem, LTTextContainer):
                                    # Check if text is inside or near the rectangle
                                    if (element.x0 <= text_elem.x0 <= element.x1 and 
                                        element.y0 <= text_elem.y0 <= element.y1):
                                        tables.append({
                                            'x0': element.x0,
                                            'y0': element.y0,
                                            'x1': element.x1,
                                            'y1': element.y1,
                                            'text': text_elem.get_text().strip()
                                        })
                
                # Combine all text while attempting to preserve structure
                text = "\n\n".join(paragraphs)
                
                result[page_number] = {
                    'text': text,
                    'tables': tables,
                    'paragraphs': paragraphs,
                    'metadata': {
                        'page_number': page_number
                    }
                }
        except Exception as e:
            print(f"Error extracting with PDFMiner: {e}")
            # Fall back to empty content
            return {1: {'text': '', 'metadata': {'error': str(e)}}}
        
        return result
    
    def _extract_with_ocr(self, pdf_path: str) -> Dict[int, Dict]:
        """
        Extract text using OCR for scanned documents.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with page numbers as keys and OCR-extracted content
        """
        result = {}
        
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            
            for i, image in enumerate(images):
                page_number = i + 1
                
                # Use pytesseract for OCR
                text = pytesseract.image_to_string(image)
                
                # Use image_to_data for more structured extraction
                ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                
                # Simple paragraph detection based on line spacing
                paragraphs = []
                current_paragraph = []
                last_top = -100
                
                for j, word in enumerate(ocr_data['text']):
                    if word.strip():
                        current_top = ocr_data['top'][j]
                        
                        # If significant vertical space, start new paragraph
                        if current_top - last_top > 25:  # Threshold for new paragraph
                            if current_paragraph:
                                paragraphs.append(' '.join(current_paragraph))
                                current_paragraph = []
                        
                        current_paragraph.append(word)
                        last_top = current_top
                
                # Add the last paragraph
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                
                result[page_number] = {
                    'text': text,
                    'paragraphs': paragraphs,
                    'metadata': {
                        'page_number': page_number,
                        'extraction_method': 'ocr'
                    }
                }
        except Exception as e:
            print(f"Error extracting with OCR: {e}")
            # Return minimal content on error
            return {1: {'text': '', 'metadata': {'error': str(e)}}}
        
        return result
    
    def extract_tables(self, pdf_path: str) -> Dict[int, List[Dict]]:
        """
        Extract tables from PDF. This is a simplified implementation.
        For production use, consider using libraries like camelot-py.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with page numbers as keys and lists of tables as values
        """
        # This is a placeholder for table extraction
        # In a real implementation, you would use specialized libraries
        # like camelot-py or tabula-py
        result = {}
        
        # For now, we'll return the simple table detection from pdfminer
        pdfminer_result = self._extract_with_pdfminer(pdf_path)
        
        for page_num, page_content in pdfminer_result.items():
            if 'tables' in page_content and page_content['tables']:
                result[page_num] = page_content['tables']
            else:
                result[page_num] = []
        
        return result

    def get_document_metadata(self, pdf_path: str) -> Dict:
        """
        Extract document metadata like title, author, etc.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with metadata fields
        """
        metadata = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                if reader.metadata:
                    # Convert from PyPDF2's DocumentInformation to dict
                    for key, value in reader.metadata.items():
                        if key.startswith('/'):
                            clean_key = key[1:]  # Remove leading slash
                        else:
                            clean_key = key
                        metadata[clean_key] = value
                
                # Add basic document stats
                metadata['pages'] = len(reader.pages)
                
                # Try to detect if document is encrypted
                metadata['encrypted'] = reader.is_encrypted
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            metadata['error'] = str(e)
        
        return metadata