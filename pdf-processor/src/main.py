"""
PDF processor service - gRPC server implementation for PDF processing with PII redaction
and translation capabilities.
"""
import os
import uuid
import logging
import concurrent.futures
import json
import datetime
import tempfile
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import grpc
import re
import signal
import sys

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generated import processor_pb2 as pdf_processor_pb2
from generated import processor_pb2_grpc as pdf_processor_pb2_grpc

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup directories
INPUT_DIR = Path("/app/data/input")
OUTPUT_DIR = Path("/app/data/output")
TEMP_DIR = Path("/app/data/temp")
JOB_DIR = Path("/app/data/jobs")  # New directory for job tracking

# Ensure directories exist
for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR, JOB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class JobManager:
    """Manages processing jobs, tracking status and results"""
    
    def __init__(self):
        """Initialize the job manager"""
        self.cleanup_interval = 24 * 60 * 60  # 24 hours
        self.job_retention_days = 7
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._job_cache = {}  # In-memory cache for faster lookups
        
        # Start cleanup task
        self._executor.submit(self._cleanup_old_jobs)
        
    def create_job(self, job_id: str, original_filename: str) -> Dict:
        """Create a new job record"""
        job_data = {
            "job_id": job_id,
            "status": "created",
            "original_filename": original_filename,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "metadata": {}
        }
        
        # Save to disk
        self._save_job_data(job_id, job_data)
        
        # Add to cache
        self._job_cache[job_id] = job_data
        
        return job_data
        
    def update_job(self, job_id: str, status: str, metadata: Optional[Dict] = None, 
                  output_file: Optional[str] = None, error: Optional[str] = None) -> Dict:
        """Update a job's status and details"""
        job_data = self.get_job(job_id)
        
        if not job_data:
            # Create a new job if it doesn't exist
            job_data = {
                "job_id": job_id,
                "status": status,
                "created_at": datetime.datetime.now().isoformat(),
                "metadata": {}
            }
        
        # Update the job
        job_data["status"] = status
        job_data["updated_at"] = datetime.datetime.now().isoformat()
        
        if metadata:
            if "metadata" not in job_data:
                job_data["metadata"] = {}
            job_data["metadata"].update(metadata)
            
        if output_file:
            job_data["output_file"] = output_file
            
        if error:
            job_data["error"] = error
            
        # Save to disk
        self._save_job_data(job_id, job_data)
        
        # Update cache
        self._job_cache[job_id] = job_data
        
        return job_data
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job status and details"""
        # Check cache first
        if job_id in self._job_cache:
            return self._job_cache[job_id]
            
        # Check disk
        job_file = JOB_DIR / f"{job_id}.json"
        if job_file.exists():
            try:
                with open(job_file, "r") as f:
                    job_data = json.load(f)
                    # Update cache
                    self._job_cache[job_id] = job_data
                    return job_data
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error reading job file for {job_id}: {str(e)}")
                
        return None
    
    def _save_job_data(self, job_id: str, job_data: Dict) -> None:
        """Save job data to disk"""
        job_file = JOB_DIR / f"{job_id}.json"
        try:
            with open(job_file, "w") as f:
                json.dump(job_data, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving job data for {job_id}: {str(e)}")
            
    def _cleanup_old_jobs(self) -> None:
        """Periodically clean up old job files"""
        while True:
            try:
                logger.info("Starting job cleanup")
                cutoff_time = datetime.datetime.now() - datetime.timedelta(days=self.job_retention_days)
                cutoff_str = cutoff_time.isoformat()
                
                # Clean up job files
                for job_file in JOB_DIR.glob("*.json"):
                    try:
                        with open(job_file, "r") as f:
                            job_data = json.load(f)
                            
                        if job_data.get("updated_at", "") < cutoff_str:
                            # Delete the job file
                            job_file.unlink()
                            
                            # Remove from cache
                            job_id = job_data.get("job_id")
                            if job_id and job_id in self._job_cache:
                                del self._job_cache[job_id]
                                
                            logger.info(f"Cleaned up old job {job_id}")
                    except Exception as e:
                        logger.error(f"Error processing job file {job_file.name}: {str(e)}")
                        
                logger.info("Job cleanup completed")
            except Exception as e:
                logger.error(f"Error in job cleanup: {str(e)}")
                
            # Sleep until next cleanup
            time.sleep(self.cleanup_interval)


class PDFProcessorServicer(pdf_processor_pb2_grpc.PDFProcessorServicer):
    """
    Implementation of PDFProcessor service with optimized performance and
    resource management.
    """

    def __init__(self):
        """Initialize the service with lazy loading of components"""
        self._pdf_extractor = None
        self._pii_detector = None
        self._job_manager = JobManager()
        
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickling issues
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"PDF Processor service initialized with ThreadPoolExecutor")

    @property
    def pdf_extractor(self):
        """Lazy load the PDF extractor"""
        if self._pdf_extractor is None:
            from src.pdf_extractor.pdf_extractor import PDFExtractor
            self._pdf_extractor = PDFExtractor(
                enable_ocr=os.getenv('ENABLE_OCR', 'true').lower() == 'true'
            )
            logger.info("Initialized PDFExtractor")
        return self._pdf_extractor

    @property
    def pii_detector(self):
        """Lazy load the PII detector"""
        if self._pii_detector is None:
            from src.pii_detector.pii_detector import PIIDetector
            self._pii_detector = PIIDetector(enable_multithreading=False)  # Disable threading to avoid pickle issues
            logger.info("Initialized PIIDetector")
        return self._pii_detector

    def ProcessDocument(self, request, context):
        """
        Process a PDF document with enhanced error handling and asynchronous processing.
        
        Args:
            request: The ProcessRequest message
            context: The gRPC context
            
        Returns:
            ProcessResponse message with job ID and metadata
        """
        try:
            # Generate a unique job ID
            job_id = str(uuid.uuid4())
            logger.info(f"Processing document {request.filename} with job_id {job_id}")

            # Create job record
            self._job_manager.create_job(job_id, request.filename)

            # Save the uploaded file
            input_file_path = INPUT_DIR / f"{job_id}_{request.filename}"

            # Write incoming document to file
            with open(input_file_path, "wb") as f:
                f.write(request.document)
                
            # Update job status
            self._job_manager.update_job(job_id, "processing")

            # Start background processing
            self._executor.submit(
                self._process_document_background, 
                job_id, 
                input_file_path, 
                request.filename,
                request.options
            )

            # Create basic response
            response = pdf_processor_pb2.ProcessResponse(
                job_id=job_id,
                original_filename=request.filename
            )
            
            # Add metadata about queued processing
            response.metadata["status"] = "processing"

            logger.info(f"Queued job {job_id} for background processing")
            return response

        except Exception as e:
            logger.error(f"Error initiating document processing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error initiating document processing: {str(e)}")
            return pdf_processor_pb2.ProcessResponse()

    def _process_document_background(self, job_id, input_file_path, filename, options):
        """Process a document in the background with comprehensive error handling."""
        output_file_path = None
        try:
            logger.info(f"Starting background processing for job {job_id}")
            output_file_path = OUTPUT_DIR / f"{job_id}_{filename}"

            # Extract text from PDF
            logger.info(f"Extracting text from {input_file_path}")
            document = self.pdf_extractor.extract_text(str(input_file_path))

            # Get document metadata
            metadata = self.pdf_extractor.get_document_metadata(str(input_file_path))

            # Determine document language from first page text
            first_page_text = ""
            for page_num in sorted(document.keys()):
                if 'text' in document[page_num] and document[page_num]['text']:
                    first_page_text = document[page_num]['text']
                    break

            # Simple language detection
            language = self._detect_language(first_page_text)
            logger.info(f"Detected document language: {language}")

            # Update job with metadata
            self._job_manager.update_job(job_id, "processing", metadata={
                "page_count": metadata.get('pages', 0),
                "extraction_complete": "true",
                "language": language
            })

            # Create a copy of the original document for processing
            processed_document = document.copy()

            # Apply redaction if requested
            redaction_stats = None
            if options.enable_redaction:
                redaction_stats = self._apply_redaction(
                    job_id, processed_document, options, input_file_path,
                    output_file_path, language
                )

            # Apply translation if requested (after redaction)
            if options.enable_translation:
                translation_success = self._apply_translation(
                    job_id, processed_document, options, output_file_path
                )
                if not translation_success:
                    raise RuntimeError("Translation failed")

            # If no processing was requested, just copy the file
            if not options.enable_translation and not options.enable_redaction:
                logger.info(f"No processing requested, copying file for job {job_id}")
                with open(input_file_path, "rb") as src, open(output_file_path, "wb") as dst:
                    dst.write(src.read())
                    
            # Update job status to completed
            self._job_manager.update_job(job_id, "completed", output_file=output_file_path.name)
            
            # Update the global response with redaction stats
            if redaction_stats:
                self._update_job_with_redaction_stats(job_id, redaction_stats)
                
            logger.info(f"Successfully completed background processing for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error in background processing for job {job_id}: {str(e)}")
            import traceback
            error_details = traceback.format_exc()
            logger.error(error_details)
            
            # Save error log
            error_log_path = OUTPUT_DIR / f"{job_id}_error.log"
            with open(error_log_path, "w") as f:
                f.write(f"Error processing document: {str(e)}\n\n")
                f.write(error_details)
                
            # Update job status
            self._job_manager.update_job(job_id, "failed", error=str(e))
            
            # If we have a valid input file but processing failed, copy it to output
            if input_file_path.exists() and (output_file_path is None or not output_file_path.exists()):
                try:
                    output_file_path = OUTPUT_DIR / f"{job_id}_FAILED_{filename}"
                    with open(input_file_path, "rb") as src, open(output_file_path, "wb") as dst:
                        dst.write(src.read())
                    self._job_manager.update_job(job_id, "failed", output_file=output_file_path.name)
                except Exception as copy_error:
                    logger.error(f"Error copying original file after failure: {str(copy_error)}")

    def _detect_language(self, text: str) -> str:
        """
        Simple language detection for Indonesian vs English.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code ('en' or 'id')
        """
        # Indonesian common words that strongly indicate the language
        indonesian_indicators = [
            "dan", "yang", "di", "ini", "dengan", "untuk", "tidak", "dalam",
            "adalah", "pada", "akan", "dari", "telah", "oleh", "atau", "juga",
            "ke", "karena", "tersebut", "bisa", "ada", "mereka", "lebih", "tahun",
            "sudah", "saya", "kita", "seperti", "kami", "kepada", "hanya", "banyak",
            "sebagai", "jalan", "nomor", "satu", "dua", "tiga", "empat", "lima"
        ]

        # Count Indonesian words in the text
        text_lower = text.lower()
        indonesian_count = sum(1 for word in indonesian_indicators if f" {word} " in f" {text_lower} ")

        # If more than 5 Indonesian indicator words are found, consider it Indonesian
        if indonesian_count > 5:
            logger.info(f"Detected Indonesian language with {indonesian_count} indicator words")
            return "id"
        return "en"

    def _apply_redaction(self, job_id, document, options, input_file_path, output_file_path, language="en"):
        """
        Apply PII redaction to the document with error handling and fallback strategies.
        """
        # Check if redaction is enabled
        if not options.enable_redaction:
            return None

        # Determine what to redact
        redaction_types = list(options.redaction_types) if options.redaction_types else None

        # Update job status
        self._job_manager.update_job(job_id, "detecting_pii", metadata={
            "redaction_types": ",".join(redaction_types) if redaction_types else "all"
        })

        try:
            # Create a language-appropriate PII detector
            detector = self.pii_detector

            # Detect and redact PII in the document
            redacted_document, redacted_items = detector.redact_document(
                document, redaction_types
            )

            logger.info(f"Detected {len(redacted_items)} items to redact for job {job_id}")

            # Update job status
            self._job_manager.update_job(job_id, "redacting", metadata={
                "pii_detected": str(len(redacted_items))
            })

            # If there are items to redact, try to apply them to the PDF
            redaction_success = False
            if redacted_items:
                # Try PyMuPDF redaction
                logger.info(f"Attempting redaction with PyMuPDF for job {job_id}")
                redaction_success = self._redact_pdf_with_pymupdf(
                    str(input_file_path), str(output_file_path), redacted_items
                )

                # If PyMuPDF fails, try pdf-redactor
                if not redaction_success:
                    logger.warning(f"PyMuPDF redaction failed for job {job_id}, trying pdf-redactor")
                    redaction_success = self._redact_pdf_with_pdf_redactor(
                        str(input_file_path), str(output_file_path), redacted_items
                    )
            else:
                # No items to redact
                logger.info(f"No PII found to redact for job {job_id}")
                redaction_success = True
                # Just copy the original file
                with open(input_file_path, "rb") as src, open(output_file_path, "wb") as dst:
                    dst.write(src.read())

            # Get redaction statistics
            stats = self.pii_detector.get_pii_statistics(job_id, document, redaction_types)

            # Handle redaction results
            if not redaction_success and len(redacted_items) > 0:
                logger.error(f"All redaction methods failed for job {job_id}")
                self._job_manager.update_job(job_id, "redaction_failed", metadata={
                    "redaction_status": "FAILED",
                    "warning": f"Document redaction failed; found {len(redacted_items)} items that could not be redacted."
                })
                # Copy the original file as the output (with warning)
                with open(input_file_path, "rb") as src, open(output_file_path, "wb") as dst:
                    dst.write(src.read())
            else:
                # Successful redaction or no PII found
                if len(redacted_items) > 0:
                    self._job_manager.update_job(job_id, "redacted", metadata={
                        "redaction_status": "SUCCESS"
                    })
                    logger.info(f"Successfully redacted document for job {job_id} with {len(redacted_items)} items")
                else:
                    self._job_manager.update_job(job_id, "redacted", metadata={
                        "redaction_status": "NO_PII_FOUND"
                    })
                    logger.info(f"No PII found in document for job {job_id}")

            return stats

        except Exception as e:
            logger.error(f"Error redacting document for job {job_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self._job_manager.update_job(job_id, "redaction_error", metadata={
                "redaction_error": str(e)
            })
            return None

    def _apply_translation(self, job_id, document, options, output_file_path):
        """
        Apply translation to the document - placeholder for now.
        """
        if not options.enable_translation:
            return True
            
        logger.warning("Translation feature is currently disabled")
        return False

    def _redact_pdf_with_pymupdf(self, input_path, output_path, redacted_items):
        """Apply redactions to a PDF file with improved handling for different languages."""
        try:
            import fitz  # PyMuPDF

            # Open the PDF
            doc = fitz.open(input_path)

            # Group redactions by page and create redaction annotations
            redactions_by_page = {}

            # Group by page
            for item in redacted_items:
                if "page" in item:
                    page_num = item["page"]
                    if page_num not in redactions_by_page:
                        redactions_by_page[page_num] = []
                    redactions_by_page[page_num].append(item)

            # Process redactions page by page
            for page_num, items in redactions_by_page.items():
                try:
                    # Adjust for 0-based indexing in PyMuPDF
                    page_idx = page_num - 1
                    if page_idx < 0 or page_idx >= len(doc):
                        logger.warning(f"Page {page_num} out of range (document has {len(doc)} pages)")
                        continue

                    page = doc[page_idx]

                    # Apply redactions to each item
                    for item in items:
                        if 'original' not in item:
                            continue

                        text_to_redact = item['original']
                        
                        # Try to find and redact the text
                        success = self._redact_text_on_page(page, text_to_redact)
                        if success:
                            logger.debug(f"Successfully redacted '{text_to_redact[:20]}...' on page {page_num}")

                    # Apply all redactions to this page
                    page.apply_redactions()

                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {str(e)}")

            # Save the redacted document
            doc.save(output_path)
            doc.close()
            return True

        except Exception as e:
            logger.error(f"Error in PDF redaction: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _redact_text_on_page(self, page, text_to_redact):
        """Try multiple strategies to redact text on a page."""
        try:
            # Strategy 1: Direct text search
            text_instances = page.search_for(text_to_redact)
            if text_instances:
                for inst in text_instances:
                    page.add_redact_annot(inst, fill=(0, 0, 0))
                return True

            # Strategy 2: Search for parts of the text
            if len(text_to_redact) > 10:
                # Try first half
                first_half = text_to_redact[:len(text_to_redact)//2]
                instances = page.search_for(first_half)
                if instances:
                    for inst in instances:
                        # Extend the rectangle to cover more text
                        extended_rect = fitz.Rect(
                            inst.x0, inst.y0,
                            inst.x1 + len(text_to_redact) * 6,  # Approximate character width
                            inst.y1
                        )
                        page.add_redact_annot(extended_rect, fill=(0, 0, 0))
                    return True

            # Strategy 3: Look through text blocks
            blocks = page.get_text("blocks")
            for block in blocks:
                block_text = block[4]  # Text content is the 5th element
                if text_to_redact in block_text or text_to_redact.lower() in block_text.lower():
                    # Create a redaction rectangle covering the text block
                    rect = fitz.Rect(block[:4])  # First 4 elements are rectangle coordinates
                    page.add_redact_annot(rect, fill=(0, 0, 0))
                    return True

            return False
        except Exception as e:
            logger.error(f"Error redacting text on page: {str(e)}")
            return False

    def _redact_pdf_with_pdf_redactor(self, input_path, output_path, redacted_items):
        """Apply redactions to a PDF file using pdf-redactor library"""
        try:
            # Dynamically import the library to handle different API versions
            try:
                import pdf_redactor
                logger.info("Successfully imported pdf_redactor module")
            except ImportError:
                logger.error("pdf-redactor module not found. Please install with: pip install pdf-redactor")
                return False

            logger.info(f"Starting PDF redaction with pdf-redactor for {len(redacted_items)} items")

            # Extract just the text strings to redact
            text_to_redact = []
            for item in redacted_items:
                if "original" in item and item["original"]:
                    text_to_redact.append(item["original"])

            if not text_to_redact:
                logger.warning("No text items found to redact!")
                return False

            logger.info(f"Redacting {len(text_to_redact)} text items")

            # Handle different versions of the API
            if hasattr(pdf_redactor, 'redact'):
                # New version of the API
                options = pdf_redactor.RedactorOptions()
                options.input_stream = open(input_path, "rb")
                options.output_stream = open(output_path, "wb")
                options.text_to_redact = text_to_redact
                options.replacement_text = "██████"  # Black box
                
                pdf_redactor.redact(options)
                options.input_stream.close()
                options.output_stream.close()
            elif hasattr(pdf_redactor, 'redactor'):
                # Old version of the API
                options = {
                    "input_stream": open(input_path, "rb"),
                    "output_stream": open(output_path, "wb"),
                    "text_to_redact": text_to_redact,
                    "replacement_text": "██████"  # Black box
                }
                
                pdf_redactor.redactor(options)
                options["input_stream"].close()
                options["output_stream"].close()
            else:
                logger.error("Could not find the correct redaction function in the pdf_redactor module")
                return False

            # Verify the output file exists and has content
            if not os.path.exists(output_path):
                logger.error("Output file does not exist after redaction")
                return False

            if os.path.getsize(output_path) == 0:
                logger.error("Output file is empty after redaction")
                return False

            logger.info(f"Successfully redacted PDF: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error redacting PDF with pdf-redactor: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _update_job_with_redaction_stats(self, job_id, stats):
        """Update job with redaction statistics"""
        # Convert stats to metadata
        metadata = {
            "total_pii_count": str(stats.get('total_pii_count', 0))
        }
        
        # Add type breakdown
        type_stats = []
        for pii_type, count in stats.get('by_type', {}).items():
            type_stats.append(f"{pii_type}:{count}")
        metadata["pii_by_type"] = ",".join(type_stats)
        
        # Add method breakdown
        method_stats = []
        for method, count in stats.get('by_method', {}).items():
            method_stats.append(f"{method}:{count}")
        metadata["pii_by_method"] = ",".join(method_stats)
        
        # Update job
        self._job_manager.update_job(job_id, "completed", metadata=metadata)

    def GetStatus(self, request, context):
        """
        Get status of a processing job with improved error handling
        and detailed status reporting.
        """
        job_id = request.job_id
        logger.info(f"Checking status for job {job_id}")
        
        # Check job status from job manager
        job_data = self._job_manager.get_job(job_id)
        
        if job_data:
            # Job exists in our tracking system
            status = job_data.get("status", "unknown")
            output_file = job_data.get("output_file")
            
            if output_file:
                # Double-check that the output file exists
                output_path = OUTPUT_DIR / output_file
                if not output_path.exists():
                    # Try to find the file with any language suffix
                    possible_files = list(OUTPUT_DIR.glob(f"{job_id}_*_{output_file.split('_')[-1]}"))
                    if possible_files:
                        # Found a matching file with language suffix
                        output_file = possible_files[0].name
                        # Update job data with correct filename
                        self._job_manager.update_job(job_id, status, output_file=output_file)
                    else:
                        # File is missing, mark as error
                        logger.warning(f"Output file {output_file} for job {job_id} is missing")
                        status = "error"
                        output_file = None
                    
            return pdf_processor_pb2.StatusResponse(
                job_id=job_id,
                status=status,
                output_file=output_file or ""
            )
        
        # If job not in tracking system, check for files
        # Check for any output files with job_id prefix
        output_files = list(OUTPUT_DIR.glob(f"{job_id}_*.pdf"))
        
        if output_files:
            logger.info(f"Found output file: {output_files[0].name}")
            return pdf_processor_pb2.StatusResponse(
                job_id=job_id,
                status="completed",
                output_file=output_files[0].name
            )

        # Check if input file exists but processing not complete
        input_files = list(INPUT_DIR.glob(f"{job_id}_*"))

        if input_files:
            return pdf_processor_pb2.StatusResponse(
                job_id=job_id,
                status="processing"
            )

        # Job not found
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details(f"Job {job_id} not found")
        return pdf_processor_pb2.StatusResponse()

    def StreamPIIDetection(self, request, context):
        """
        Stream PII detection results with chunked processing for
        large documents and better error handling.
        """
        try:
            logger.info(f"Starting PII detection stream for {request.filename}")
            
            # Use a unique temp file for this request
            with tempfile.NamedTemporaryFile(suffix=f"_{request.filename}", delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(request.document)
                
            logger.info(f"Saved document to temporary file: {temp_file_path}")
            
            try:
                # Extract text
                document = self.pdf_extractor.extract_text(temp_file_path)
                
                # Process each page
                for page_num, page_content in document.items():
                    if 'text' in page_content:
                        # For large pages, process in chunks
                        if len(page_content['text']) > 50000:  # ~10 pages of text
                            logger.info(f"Processing large page {page_num} in chunks")
                            # Process text in chunks to avoid memory issues
                            pii_instances = self.pii_detector.detect_pii_in_chunks(page_content['text'])
                        else:
                            # Regular detection for smaller pages
                            pii_instances = self.pii_detector.detect_pii(page_content['text'])
                        
                        # Stream each PII instance
                        for pii in pii_instances:
                            yield pdf_processor_pb2.PIIDetectionResult(
                                pii_text=pii['text'],
                                pii_type=pii['type'],
                                page=page_num,
                                start=pii['start'],
                                end=pii['end']
                            )
                            
                        # Log progress
                        logger.info(f"Streamed {len(pii_instances)} PII instances from page {page_num}")
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                    logger.info(f"Removed temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Error removing temporary file: {str(e)}")
                    
            logger.info(f"Completed PII detection stream for {request.filename}")

        except Exception as e:
            logger.error(f"Error in PII detection stream: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error in PII detection: {str(e)}")


def serve():
    """Start the gRPC server with health checking and graceful shutdown"""
    # Create server with more workers for better concurrency
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=10)
    )
    
    # Register service
    pdf_processor_pb2_grpc.add_PDFProcessorServicer_to_server(
        PDFProcessorServicer(), server
    )
    
    # Add health checking
    try:
        from grpc_health.v1 import health_pb2_grpc, health
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
        logger.info("Added health checking service")
    except ImportError:
        logger.warning("Health checking not available - grpc_health not installed")
    
    # Configure server address
    server_address = '[::]:50051'
    server.add_insecure_port(server_address)
    
    # Handle graceful shutdown
    def handle_sigterm(*args):
        logger.info("Received shutdown signal, stopping server...")
        stopped_event = server.stop(30)  # 30 seconds grace period
        stopped_event.wait(45)  # Wait up to 45 seconds for completion
        logger.info("Server stopped")
        sys.exit(0)
        
    # Register signal handlers for graceful shutdown
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        signal.signal(sig, handle_sigterm)
    
    # Start server
    server.start()
    logger.info(f"Server started, listening on {server_address}")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(60 * 60 * 24)  # Sleep for a day
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping server...")
        server.stop(0)


if __name__ == '__main__':
    serve()