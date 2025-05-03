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
import multiprocessing

import sys
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
    
        self.max_workers = min(4, cpu_count * 2)

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        logger.info(f"PDF Processor service initialized with {self.max_workers} workers")

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
            
    def _process_document_background(self, job_id, input_file_path, filename, options):
        """
        Process a document in the background with comprehensive error handling.
        
        Args:
            job_id: Unique job identifier
            input_file_path: Path to the input PDF file
            filename: Original filename
            options: Processing options from the request
        """
        output_file_path = None
        try:
            logger.info(f"Starting background processing for job {job_id}")
            output_file_path = OUTPUT_DIR / f"{job_id}_{filename}"
            
            # Extract text from PDF
            logger.info(f"Extracting text from {input_file_path}")
            document = self.pdf_extractor.extract_text(str(input_file_path))
            
            # Get document metadata
            metadata = self.pdf_extractor.get_document_metadata(str(input_file_path))
            
            # Update job with metadata
            self._job_manager.update_job(job_id, "processing", metadata={
                "page_count": metadata.get('pages', 0),
                "extraction_complete": "true"
            })
            
            # Create a copy of the original document for processing
            processed_document = document.copy()
            
            # Apply translation if requested
            if options.enable_translation:
                self._apply_translation(
                    job_id, processed_document, options, output_file_path
                )
            
            # Apply PII redaction if requested
            redaction_stats = None
            if options.enable_redaction:
                redaction_stats = self._apply_redaction(
                    job_id, processed_document, options, input_file_path, output_file_path
                )
            
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
            # This ensures the user can still download the original file
            if input_file_path.exists() and (output_file_path is None or not output_file_path.exists()):
                try:
                    output_file_path = OUTPUT_DIR / f"{job_id}_FAILED_{filename}"
                    with open(input_file_path, "rb") as src, open(output_file_path, "wb") as dst:
                        dst.write(src.read())
                    self._job_manager.update_job(job_id, "failed", output_file=output_file_path.name)
                except Exception as copy_error:
                    logger.error(f"Error copying original file after failure: {str(copy_error)}")
        
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

        # Translate the document
        try:
            # This will modify the document in-place
            translated_document = translation_service.translate_document(
                document,
                source_language,
                target_language
            )
            
            # Update job status
            self._job_manager.update_job(job_id, "translated", metadata={
                "translated": "true"
            })
            
            # Modify the output filename to indicate translation
            output_lang_suffix = target_language.split('_')[0] if '_' in target_language else target_language
            new_output_path = OUTPUT_DIR / f"{job_id}_{output_lang_suffix}_{os.path.basename(output_file_path)}"
            output_file_path.rename(new_output_path)
            
            return True
        except Exception as e:
            logger.error(f"Error during translation: {str(e)}")
            self._job_manager.update_job(job_id, "translation_failed", metadata={
                "translation_error": str(e)
            })
            return False
            
    def _apply_redaction(self, job_id, document, options, input_file_path, output_file_path):
        """
        Apply PII redaction to the document with error handling and fallback strategies.
        
        Args:
            job_id: Unique job identifier
            document: Document to redact
            options: Processing options
            input_file_path: Path to the input file
            output_file_path: Path where to save the output file
            
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
            # Detect and redact PII in the text representation
            redacted_document, redacted_items = self.pii_detector.redact_document(
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
                    # Try PyMuPDF first (more robust)
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
        Create a new PDF document from translated text with better
        formatting preservation.
        
        Args:
            document: Document dictionary with translated text
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

                # Add a new page
                page = doc.new_page()

                # Better rendering with paragraph structure
                if 'paragraphs' in page_content and page_content['paragraphs']:
                    # Calculate page margins
                    margin = 50  # points
                    content_width = page.rect.width - 2 * margin
                    
                    # Initial y position
                    y_pos = margin
                    
                    # Process each paragraph
                    for paragraph in page_content['paragraphs']:
                        if not paragraph.strip():
                            # Empty paragraph - just add some space
                            y_pos += 10
                            continue
                            
                        # Insert the paragraph text
                        text_rect = fitz.Rect(margin, y_pos, margin + content_width, y_pos + 500)
                        text_height = page.insert_textbox(
                            text_rect,
                            paragraph,
                            fontname="helv",
                            fontsize=11,
                            align=0  # 0=left, 1=center, 2=right
                        )
                        
                        # Move y position down
                        y_pos += text_height + 10  # Add some spacing between paragraphs
                        
                        # If we're close to the bottom, start a new page
                        if y_pos > page.rect.height - margin:
                            page = doc.new_page()
                            y_pos = margin
                else:
                    # Simple rendering - using just the main text
                    if 'text' in page_content and page_content['text']:
                        text = page_content['text']

                        # Create a text block that fits on the page
                        rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
                        page.insert_text(
                            fitz.Point(50, 50),  # Starting position
                            text,
                            fontname="helv",  # A standard font
                            fontsize=11,
                            rect=rect
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
        """
        Apply redactions to a PDF file using PyMuPDF (fitz) with improved
        redaction accuracy and flexible fallbacks.
        """
        try:
            import fitz  # PyMuPDF

            logger.info(f"Starting PDF redaction with PyMuPDF for {len(redacted_items)} items")

            # Open the PDF
            doc = fitz.open(input_path)

            # Track if we made any changes
            redactions_applied = False

            # Group redactions by page
            redactions_by_page = {}
            for item in redacted_items:
                if "page" in item and "original" in item:
                    page_num = item["page"]
                    if page_num not in redactions_by_page:
                        redactions_by_page[page_num] = []
                    redactions_by_page[page_num].append(item)

            logger.info(f"Processing redactions across {len(redactions_by_page)} pages")

            cc_items = [item for item in redacted_items if 'type' in item and item['type'] == 'credit_card']
            logger.info(f"Found {len(cc_items)} credit card items to redact")
        
            for item in cc_items:
                logger.info(f"Credit card to redact: {item}")

            # Process each page
            for page_num, items in redactions_by_page.items():
                try:
                    # Adjust for 0-based indexing in PyMuPDF
                    page_idx = page_num - 1
                    if page_idx < 0 or page_idx >= len(doc):
                        logger.warning(f"Page {page_num} out of range (document has {len(doc)} pages)")
                        continue

                    page = doc[page_idx]
                    
                    # Redaction approaches in order of preference
                    redaction_approaches = [
                        self._redact_with_exact_search,
                        self._redact_with_fuzzy_search,
                        self._redact_with_block_scan,
                        self._redact_with_page_dict_scan
                    ]
                    
                    # Try each approach for the items on this page
                    for item in items:
                        text_to_find = item["original"]
                        
                        # Skip very short strings
                        if len(text_to_find.strip()) < 3:
                            logger.warning(f"Skipping very short text for redaction: '{text_to_find}'")
                            continue
                            
                        # Try each approach in order
                        item_redacted = False
                        for approach in redaction_approaches:
                            try:
                                approach_result = approach(page, text_to_find)
                                if approach_result:
                                    item_redacted = True
                                    redactions_applied = True
                                    break
                            except Exception as approach_error:
                                logger.warning(f"Redaction approach failed: {str(approach_error)}")
                                continue
                                
                        if not item_redacted:
                            logger.warning(f"Could not redact '{text_to_find[:20]}...' with any approach")

                    # Apply the redactions to this page
                    page.apply_redactions()
                    logger.info(f"Applied redactions to page {page_num}")

                    # TODO REMOVE LATER
                    cc_items_for_page = [item for item in items if 'type' in item and item['type'] == 'credit_card']
                    if cc_items_for_page:
                        logger.info(f"Processing {len(cc_items_for_page)} credit card redactions on page {page_num}")
                        
                        # After text search for each credit card
                        for item in cc_items_for_page:
                            text_to_find = item["original"]
                            text_instances = page.search_for(text_to_find)
                            logger.info(f"Credit card search results: {text_instances} for text: {text_to_find}")

                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {str(e)}")
                    continue

            # Save the document if any redactions were applied
            if redactions_applied:
                logger.info(f"Saving redacted document to {output_path}")
                doc.save(output_path)
                doc.close()
                return True
            else:
                logger.warning("No redactions were applied to the document")
                doc.close()
                return False

        except Exception as e:
            logger.error(f"Error redacting PDF with PyMuPDF: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
        # Check specifically for a PDF output file first
        pdf_output_files = list(OUTPUT_DIR.glob(f"{job_id}_*.pdf"))

        if pdf_output_files:
            logger.info(f"Found PDF output file: {pdf_output_files[0].name}")
            return pdf_processor_pb2.StatusResponse(
                job_id=job_id,
                status="completed",
                output_file=pdf_output_files[0].name
            )

        # If no PDF file, check for other output files
        other_output_files = [f for f in OUTPUT_DIR.glob(f"{job_id}_*")
                              if not f.name.endswith('_log.txt') and not f.name.endswith('_REDACTION_FAILED.txt')]

        if other_output_files:
            logger.info(f"Found non-PDF output file: {other_output_files[0].name}")
            return pdf_processor_pb2.StatusResponse(
                job_id=job_id,
                status="completed",
                output_file=other_output_files[0].name
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
                        
                        for i, pattern in enumerate(patterns):
                            matches = re.findall(pattern, page_content['text'])
                            if matches:
                                logger.info(f"Page {page_num} - Found potential credit card with pattern {i}: {matches}")

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