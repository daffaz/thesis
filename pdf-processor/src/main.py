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
import multiprocessing

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generated import processor_pb2 as pdf_processor_pb2
from generated import processor_pb2_grpc as pdf_processor_pb2_grpc
from src.pii_detector.pii_detector import PIIDetector

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
                            if job_id in self._job_cache:
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
        self._translation_service = None
        self._job_manager = JobManager()

        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
        except (ImportError, NotImplementedError):
            cpu_count = 2  # Fallback if cpu_count() isn't available
    
        # Limit max workers to avoid CPU overload
        self.max_workers = min(max(2, cpu_count - 1), 4)  # Leave 1 CPU free, max 4 workers
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Create a process pool for CPU-intensive tasks
        self._process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, cpu_count - 1)  # Leave 1 CPU free
        )
        
        logger.info(f"PDF Processor service initialized with {self.max_workers} thread workers and {cpu_count-1} process workers")

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
            self._pii_detector = PIIDetector(
                enable_multithreading=True
            )
            logger.info("Initialized PIIDetector")
        return self._pii_detector

    def get_translation_service(self, force_reload=False):
        """Get or initialize the translation service"""
        if self._translation_service is None or force_reload:
            try:
                from src.translation.translation import TranslationService
                model_path = os.getenv('TRANSLATION_MODEL_PATH')
                self._translation_service = TranslationService(model_path)
                logger.info(f"Initialized TranslationService with model from {model_path or 'HuggingFace'}")
            except Exception as e:
                logger.error(f"Error initializing translation service: {str(e)}")
                return None
        return self._translation_service

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

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the text.
        Currently supports English and Indonesian.

        Args:
            text: Text to analyze

        Returns:
            Language code ('en' or 'id')
        """
        # Simple detection based on common Indonesian words
        indonesian_words = [
            "dan", "yang", "di", "ini", "dengan", "untuk", "tidak", "dalam",
            "adalah", "pada", "akan", "dari", "telah", "oleh", "atau", "juga",
            "ke", "karena", "tersebut", "bisa", "ada", "mereka", "lebih", "tahun",
            "sudah", "saya", "kita", "seperti", "kami", "kepada", "hanya", "banyak",
            "sebagai", "jalan", "nomor", "satu", "dua", "tiga", "empat", "lima"
        ]

        # Count Indonesian words in the text
        text_lower = text.lower()
        indonesian_word_count = sum(1 for word in indonesian_words if f" {word} " in f" {text_lower} ")

        # If more than 5 Indonesian words found, consider it Indonesian
        # This is a simple heuristic - consider using langdetect library for better results
        if indonesian_word_count > 5:
            return "id"
        return "en"

    def _process_document_background(self, job_id, input_file_path, filename, options):
        """Process a document in the background with comprehensive error handling."""
        output_file_path = None
        try:
            logger.info(f"Starting background processing for job {job_id}")
            output_file_path = OUTPUT_DIR / f"{job_id}_{filename}"

            # Extract text from PDF using process pool for CPU-intensive work
            logger.info(f"Extracting text from {input_file_path}")
            future = self._process_pool.submit(self.pdf_extractor.extract_text, str(input_file_path))
            document = future.result(timeout=300)  # 5 minute timeout

            # Get document metadata
            metadata = self.pdf_extractor.get_document_metadata(str(input_file_path))

            # Determine document language from first page text
            first_page_text = ""
            for page_num in sorted(document.keys()):
                if 'text' in document[page_num] and document[page_num]['text']:
                    first_page_text = document[page_num]['text']
                    break

            # Import the language detection function
            from src.utils import detect_document_language

            # Detect language using process pool
            future = self._process_pool.submit(detect_document_language, first_page_text)
            language = future.result(timeout=60)
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
                # Create a language-appropriate PII detector
                pii_detector = PIIDetector(language=language)

                # Apply redaction with language-specific detection using process pool
                future = self._process_pool.submit(
                    self._apply_redaction,
                    job_id, processed_document, options, input_file_path,
                    output_file_path, pii_detector, language
                )
                redaction_stats = future.result(timeout=300)  # 5 minute timeout

            # Apply translation if requested (after the other changes)
            if options.enable_translation:
                future = self._process_pool.submit(
                    self._apply_translation,
                    job_id, processed_document, options, output_file_path
                )
                translation_success = future.result(timeout=300)  # 5 minute timeout
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
            
        except concurrent.futures.TimeoutError:
            logger.error(f"Processing timeout for job {job_id}")
            self._job_manager.update_job(job_id, "failed", error="Processing timeout")
            
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
        finally:
            # Clean up any temporary files if needed
            pass

    def _apply_translation(self, job_id, document, options, output_file_path):
        """
        Apply translation to the document with error handling.
        
        Args:
            job_id: Unique job identifier
            document: Document to translate (will be modified in-place)
            options: Processing options
            output_file_path: Path where to save the output file
            
        Returns:
            True if translation was successful, False otherwise
        """
        # Check if translation is enabled
        if not options.enable_translation:
            return True
            
        source_language = options.source_language if options.source_language else None
        target_language = options.target_language

        if not target_language:
            logger.error("Target language is required for translation")
            raise ValueError("Target language is required for translation")

        logger.info(f"Translating document from {source_language or 'auto-detected'} to {target_language}")
        
        # Update job status
        self._job_manager.update_job(job_id, "translating", metadata={
            "source_language": source_language or "auto-detect",
            "target_language": target_language
        })

        # Initialize translation service
        translation_service = self.get_translation_service()
        if translation_service is None:
            logger.error("Translation service not available")
            raise RuntimeError("Translation service not available")

        try:
            # Translate the document in memory
            translated_document = translation_service.translate_document(
                document,
                source_language,
                target_language
            )
            
            # Prepare the output paths with language suffix
            output_lang_suffix = target_language.split('_')[0] if '_' in target_language else target_language
            base_name = os.path.basename(str(output_file_path))
            # Remove any existing job_id prefix from base_name to avoid duplication
            if base_name.startswith(job_id):
                base_name = base_name[len(job_id)+1:]
            new_output_path = OUTPUT_DIR / f"{job_id}_{output_lang_suffix}_{base_name}"
            
            # Create parent directory if it doesn't exist
            new_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create the translated PDF directly at the new location
            success = self._create_translated_pdf(translated_document, str(new_output_path))
            
            if success:
                # Update job status with the correct filename that includes language suffix
                self._job_manager.update_job(job_id, "completed", metadata={
                    "translated": "true"
                }, output_file=new_output_path.name)
                return True
            else:
                raise RuntimeError("Failed to create translated PDF")
                
        except Exception as e:
            logger.error(f"Error during translation: {str(e)}")
            self._job_manager.update_job(job_id, "translation_failed", metadata={
                "translation_error": str(e)
            })
            return False

    def _apply_redaction(self, job_id, document, options, input_file_path, output_file_path, pii_detector=None, language="en"):
        """
        Apply PII redaction to the document with error handling and fallback strategies.

        Args:
            job_id: Unique job identifier
            document: Document to redact
            options: Processing options
            input_file_path: Path to the input file
            output_file_path: Path where to save the output file
            pii_detector: Optional custom PII detector for language-specific detection

        Returns:
            RedactionStats if redaction was successful, None otherwise
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
            # Use the provided PII detector or create a new one
            detector = pii_detector or self.pii_detector

            # For Indonesian documents, add special NIK detection
            if language == "id":
                # Find NIKs specifically in each page
                for page_num, page_content in document.items():
                    if 'text' in page_content and page_content['text']:
                        names = pii_detector.detect_indonesian_names(page_content['text'])
                        if names:
                            logger.info(f"Detected {len(names)} Indonesian names on page {page_num}")

                        nik_instances = detector.detect_nik_with_context(page_content['text'])
                        if nik_instances:
                            logger.info(f"Detected {len(nik_instances)} NIK instances on page {page_num}")

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
                # Try different redaction strategies
                if options.enable_translation:
                    # For translation + redaction, create a new PDF
                    redaction_success = self._create_translated_pdf(
                        redacted_document,
                        str(output_file_path)
                    )
                else:
                    # First try specific phone redaction
                    logger.info(f"Attempting phone-specific redaction for job {job_id}")
                    phone_success = self._redact_phones_specifically(
                        str(input_file_path), str(output_file_path), redacted_items
                    )

                    if not phone_success:
                        # Try PyMuPDF next (more robust)
                        logger.info(f"Attempting redaction with PyMuPDF for job {job_id}")
                        pymupdf_success = self._redact_pdf_with_pymupdf(
                            str(input_file_path), str(output_file_path), redacted_items
                        )

                        # If PyMuPDF fails, try pdf-redactor
                        if not pymupdf_success:
                            logger.warning(f"PyMuPDF redaction failed for job {job_id}, trying pdf-redactor")
                            pdf_redactor_success = self._redact_pdf_with_pdf_redactor(
                                str(input_file_path), str(output_file_path), redacted_items
                            )
                            redaction_success = pdf_redactor_success
                        else:
                            redaction_success = True
                    else:
                        redaction_success = True
            else:
                # No items to redact
                logger.info(f"No PII found to redact for job {job_id}")
                redaction_success = True  # We consider this a success case

                if options.enable_translation:
                    # For translation without redaction, create a new PDF
                    redaction_success = self._create_translated_pdf(
                        document, str(output_file_path)
                    )
                else:
                    # Just copy the original file
                    with open(input_file_path, "rb") as src, open(output_file_path, "wb") as dst:
                        dst.write(src.read())

            # Get redaction statistics
            stats = self.pii_detector.get_pii_statistics(job_id, document, redaction_types)

            # If redaction fails, provide a clear error marker
            if not redaction_success and len(redacted_items) > 0:
                logger.error(f"All redaction methods failed for job {job_id}")

                # Update job with warning
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
            self._job_manager.update_job(job_id, "redaction_error", metadata={
                "redaction_error": str(e)
            })
            return None
            
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

    def _create_translated_pdf(self, document, output_path):
        """
        Create a new PDF document from translated text with improved
        formatting preservation.
        
        Args:
            document: Document dictionary with translated text and formatting info
            output_path: Path where to save the output PDF

        Returns:
            Boolean indicating success
        """
        try:
            import fitz  # PyMuPDF
            logger.info(f"Creating new PDF with translated content at {output_path}")

            # Create a new PDF document
            doc = fitz.open()

            # Process each page
            for page_num in sorted(document.keys()):
                page_content = document[page_num]
                
                # Add a new page with same dimensions as original if available
                if 'dimensions' in page_content:
                    width, height = page_content['dimensions']
                    page = doc.new_page(width=width, height=height)
                else:
                    page = doc.new_page()

                # If we have detailed text blocks with formatting
                if 'blocks' in page_content:
                    for block in page_content['blocks']:
                        # Extract block properties
                        text = block.get('text', '')
                        font = block.get('font', 'helv')
                        fontsize = block.get('fontsize', 11)
                        color = block.get('color', (0, 0, 0))
                        bbox = block.get('bbox', None)
                        align = block.get('align', 0)  # 0=left, 1=center, 2=right
                        
                        if bbox:
                            rect = fitz.Rect(bbox)
                        else:
                            # Default positioning if no bbox
                            rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
                        
                        # Insert text with preserved formatting
                        page.insert_textbox(
                            rect,
                            text,
                            fontname=font,
                            fontsize=fontsize,
                            color=color,
                            align=align
                        )
                
                # Fallback to simpler formatting if no detailed blocks
                elif 'paragraphs' in page_content and page_content['paragraphs']:
                    y_pos = 50  # Starting position
                    for paragraph in page_content['paragraphs']:
                        if not paragraph.strip():
                            y_pos += 10
                            continue
                            
                        # Try to preserve paragraph spacing and indentation
                        indent = 50
                        if paragraph.startswith('    '):  # Check for indentation
                            indent = 70
                            
                        rect = fitz.Rect(indent, y_pos, page.rect.width - 50, y_pos + 500)
                        text_height = page.insert_textbox(
                            rect,
                            paragraph,
                            fontname="helv",
                            fontsize=11,
                            align=0
                        )
                        
                        y_pos += text_height + 12  # Consistent paragraph spacing
                        
                        if y_pos > page.rect.height - 50:
                            page = doc.new_page()
                            y_pos = 50
                
                # Last resort - simple text insertion
                elif 'text' in page_content and page_content['text']:
                    rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
                    page.insert_textbox(
                        rect,
                        page_content['text'],
                        fontname="helv",
                        fontsize=11
                    )

            # Save the document
            doc.save(output_path)
            doc.close()
            logger.info(f"Successfully created translated PDF at {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating translated PDF: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
                    logger.debug(f"Adding text to redact: '{item['original'][:30]}...' (truncated for log)")

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

                    # Use multiple approaches to redact each item
                    redaction_approaches = [
                        self._redact_with_exact_search,
                        self._redact_with_fuzzy_search,
                        self._redact_with_text_instance_search,
                        self._redact_with_block_scan
                    ]

                    for item in items:
                        if 'original' not in item:
                            continue

                        text_to_redact = item['original']

                        # Try each approach
                        item_redacted = False
                        for approach in redaction_approaches:
                            try:
                                if approach(page, text_to_redact):
                                    item_redacted = True
                                    break
                            except Exception:
                                continue

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

    def _redact_phones_specifically(self, input_path, output_path, redacted_items):
        """A specialized method to detect and redact phones in PDF documents."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(input_path)

            # Extract phone items only
            phone_items = [item for item in redacted_items
                           if item.get('type') == 'phone' or 'HP:' in str(item.get('original', ''))]

            if not phone_items:
                return False

            # Process each page
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_text = page.get_text()

                # Specific patterns for phone numbers with HP: prefix
                phone_patterns = [
                    r'HP:?\s*\+?\d{1,4}[-\s.]?\d{1,4}[-\s.]?\d{1,4}[-\s.]?\d{1,4}',
                    r'HP:?\s*\d{3,4}[-\s.]?\d{3,4}[-\s.]?\d{3,4}',
                    r'HP:?\s*\d{1,4}[-\s.]?\d{1,4}[-\s.]?\d{1,4}',
                    r'HP:?\s*\(\d{2,3}\)[-\s.]?\d{3,4}[-\s.]?\d{3,4}'
                ]

                # Find all phone numbers in this page
                for pattern in phone_patterns:
                    for match in re.finditer(pattern, page_text, re.IGNORECASE):
                        match_text = match.group()
                        # Find the text instance on the page
                        # Try exact search first
                        instances = page.search_for(match_text)
                        if not instances:
                            # Try with parts of the match
                            prefix = match_text.split()[0] if ' ' in match_text else match_text[:3]
                            instances = page.search_for(prefix)

                        # Apply redaction to each instance
                        for inst in instances:
                            # Make rectangle slightly larger
                            rect = fitz.Rect(inst.x0 - 2, inst.y0 - 2,
                                             inst.x1 + 50, inst.y1 + 2)  # Extend width to catch full number
                            page.add_redact_annot(rect, fill=(0, 0, 0))

                # Apply redactions
                page.apply_redactions()

            # Save and close
            doc.save(output_path)
            doc.close()
            return True

        except Exception as e:
            logger.error(f"Error in specific phone redaction: {str(e)}")
            return False

    def _redact_with_text_instance_search(self, page, text_to_redact):
        """More aggressive text search for redaction."""
        # Get all text instances from the page using get_text
        page_text = page.get_text("text")

        if text_to_redact in page_text:
            # Text exists but may be broken up - search using character positions
            text_instances = page.search_for(text_to_redact)

            if not text_instances and len(text_to_redact) > 5:
                # Try with partial matches for longer text
                partial_text = text_to_redact[:len(text_to_redact)//2]
                text_instances = page.search_for(partial_text)

                if text_instances:
                    # Expand the matches to cover potential full text
                    expanded_instances = []
                    for inst in text_instances:
                        expanded = fitz.Rect(inst.x0, inst.y0,
                                             inst.x0 + len(text_to_redact) * 8,  # approximate width
                                             inst.y1)
                        expanded_instances.append(expanded)
                    text_instances = expanded_instances

            # Add redaction annotations
            for inst in text_instances:
                redact_annot = page.add_redact_annot(inst, fill=(0, 0, 0))
                if redact_annot:
                    return True

        return False
            
    def _redact_with_exact_search(self, page, text_to_find):
        """Try exact text search redaction"""
        text_instances = page.search_for(text_to_find)
        if text_instances:
            # Add each instance as a redaction
            for inst in text_instances:
                # Create redaction annotation - black fill
                redact_annot = page.add_redact_annot(inst, fill=(0, 0, 0))
                if not redact_annot:
                    logger.warning(f"Failed to create redaction annotation for '{text_to_find[:20]}...'")
            return True
        return False
        
    def _redact_with_fuzzy_search(self, page, text_to_find):
        """Try fuzzy search redaction for cases with spacing/formatting differences"""
        # Remove extra spaces for fuzzy matching
        fuzzy_text = ' '.join(text_to_find.split())
        
        # Try several variants
        variants = [
            fuzzy_text,
            fuzzy_text.lower(),
            fuzzy_text.upper(),
            fuzzy_text.replace(' ', ''),
            ''.join(c for c in fuzzy_text if c.isalnum())
        ]
        
        for variant in variants:
            if variant == fuzzy_text:  # Skip the original as we already tried it
                continue
                
            text_instances = page.search_for(variant)
            if text_instances:
                # Add each instance as a redaction
                for inst in text_instances:
                    # Create redaction annotation - black fill
                    redact_annot = page.add_redact_annot(inst, fill=(0, 0, 0))
                    if redact_annot:
                        return True
        return False
        
    def _redact_with_block_scan(self, page, text_to_find):
        """Scan text blocks for the target text"""
        # Get all text blocks on the page
        blocks = page.get_text("blocks")
        
        for block in blocks:
            block_text = block[4]  # Text content is the 5th element
            
            if text_to_find in block_text or text_to_find.lower() in block_text.lower():
                logger.info(f"Found text in a block: '{text_to_find[:20]}...'")
                
                # Create a redaction rectangle covering the text block
                rect = fitz.Rect(block[:4])  # First 4 elements are rectangle coordinates
                redact_annot = page.add_redact_annot(rect, fill=(0, 0, 0))
                
                if redact_annot:
                    return True
        return False
        
    def _redact_with_page_dict_scan(self, page, text_to_find):
        """Use the page dictionary to scan for text"""
        # This is the most aggressive approach - scan the entire page
        page_dict = page.get_text("dict")
        
        # Look through all blocks
        for block in page_dict.get("blocks", []):
            # Check lines
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    
                    if text_to_find in span_text or text_to_find.lower() in span_text.lower():
                        # Found the text, create a redaction
                        bbox = fitz.Rect(span["bbox"])
                        # Add some padding
                        bbox.x0 -= 2
                        bbox.y0 -= 2
                        bbox.x1 += 2
                        bbox.y1 += 2
                        
                        redact_annot = page.add_redact_annot(bbox, fill=(0, 0, 0))
                        if redact_annot:
                            return True
        
        return False

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
                        # TODO REMOVE LATER
                        logger.info(f"Page {page_num} full text: {page_content['text']}")

                        # TODO REMOVE LATER ALSO
                        patterns = [
                            r'\d{4}[-]\d{4}[-]\d{4}[-]\d{4}',  # Dashes
                            r'\d{4}\s\d{4}\s\d{4}\s\d{4}',     # Spaces
                            r'\d{16}',                          # No separators
                            r'\d{4}.?\d{4}.?\d{4}.?\d{4}'       # Any separator
                        ]
                        
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
        concurrent.futures.ThreadPoolExecutor(
            max_workers=min(12, (multiprocessing.cpu_count() or 4) * 3)
        )
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
    # Import signal module for graceful shutdown
    import signal
    serve()