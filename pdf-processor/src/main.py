"""
grpc_server.py - gRPC server for PDF processing service
"""
import os
import uuid
import logging
import concurrent.futures
from pathlib import Path
import grpc

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generated import processor_pb2 as pdf_processor_pb2
from generated import processor_pb2_grpc as pdf_processor_pb2_grpc

# Import our core functionality
from src.pdf_extractor.pdf_extractor import PDFExtractor
from src.pii_detector.pii_detector import PIIDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup directories
INPUT_DIR = Path("/app/data/input")
OUTPUT_DIR = Path("/app/data/output")
TEMP_DIR = Path("/app/data/temp")

# Ensure directories exist
for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class PDFProcessorServicer(pdf_processor_pb2_grpc.PDFProcessorServicer):
    """Implementation of PDFProcessor service"""

    def __init__(self):
        """Initialize the service with our PDF processing components"""
        self.pdf_extractor = PDFExtractor(enable_ocr=os.getenv('ENABLE_OCR', 'true').lower() == 'true')
        self.pii_detector = PIIDetector()
        logger.info("PDF Processor service initialized")

    def ProcessDocument(self, request, context):
        """
        Process a PDF document with redaction and/or translation
        """
        try:
            # Generate a unique job ID
            job_id = str(uuid.uuid4())
            logger.info(f"Processing document {request.filename} with job_id {job_id}")

            # Save the uploaded file
            input_file_path = INPUT_DIR / f"{job_id}_{request.filename}"
            output_file_path = OUTPUT_DIR / f"{job_id}_{request.filename}"

            # Write incoming document to file
            with open(input_file_path, "wb") as f:
                f.write(request.document)

            # Extract text from PDF
            document = self.pdf_extractor.extract_text(str(input_file_path))

            # Get document metadata
            metadata = self.pdf_extractor.get_document_metadata(str(input_file_path))

            # Create response
            response = pdf_processor_pb2.ProcessResponse(
                job_id=job_id,
                original_filename=request.filename
            )

            # Convert Python dict to protobuf map
            for key, value in metadata.items():
                if isinstance(value, str):
                    response.metadata[key] = value
                else:
                    # Convert non-string values to strings
                    response.metadata[key] = str(value)

            # Apply PII redaction if requested
            if request.options.enable_redaction:
                # Determine what to redact
                redaction_types = list(request.options.redaction_types) if request.options.redaction_types else None

                # Detect and redact PII in the text representation
                redacted_document, redacted_items = self.pii_detector.redact_document(
                    document, redaction_types
                )

                logger.info(f"Detected {len(redacted_items)} items to redact")

                # If there are items to redact, try to apply them to the PDF
                if redacted_items:
                    # Try redaction with PyMuPDF first (more robust)
                    logger.info("Attempting redaction with PyMuPDF (primary method)")
                    pymupdf_success = self.redact_pdf_with_pymupdf(str(input_file_path), str(output_file_path), redacted_items)

                    # If PyMuPDF fails, try with pdf-redactor
                    if not pymupdf_success:
                        logger.warning("PyMuPDF redaction failed, trying pdf-redactor as fallback")
                        pdf_redactor_success = self.redact_pdf_with_pdf_redactor(str(input_file_path), str(output_file_path), redacted_items)
                        redaction_success = pdf_redactor_success
                    else:
                        redaction_success = True
                else:
                    # No items to redact
                    logger.info("No PII found to redact")
                    redaction_success = True  # We consider this a success case (nothing to redact)

                    # Just copy the original file as output
                    with open(input_file_path, "rb") as src, open(output_file_path, "wb") as dst:
                        dst.write(src.read())

                # Get redaction statistics
                stats = self.pii_detector.get_pii_statistics(document, redaction_types)

                # Create RedactionStats message
                redaction_stats = pdf_processor_pb2.RedactionStats(
                    total_pii_count=stats['total_pii_count']
                )

                # REMOVED: Log file generation
                # We don't create PII log files anymore

                # If redaction fails, provide a clear error marker
                if not redaction_success and len(redacted_items) > 0:
                    logger.error("All redaction methods failed")

                    # Create a failure notification
                    response.metadata["redaction_status"] = "FAILED"
                    response.metadata["warning"] = f"Document redaction failed; found {len(redacted_items)} items that could not be redacted."

                    # Copy the original file as the output (with warning)
                    with open(input_file_path, "rb") as src, open(output_file_path, "wb") as dst:
                        dst.write(src.read())
                else:
                    # Successful redaction or no PII found
                    if len(redacted_items) > 0:
                        response.metadata["redaction_status"] = "SUCCESS"
                        logger.info(f"Successfully redacted document with {len(redacted_items)} items")
                    else:
                        response.metadata["redaction_status"] = "NO_PII_FOUND"
                        logger.info("No PII found in document")

                # Fill in redaction statistics
                for key, value in stats['by_type'].items():
                    redaction_stats.by_type[key] = value

                for key, value in stats['by_method'].items():
                    redaction_stats.by_method[key] = value

                for key, value in stats['by_page'].items():
                    # Convert page number to string for protobuf map
                    redaction_stats.by_page[str(key)] = value

                response.redaction_stats.CopyFrom(redaction_stats)
            else:
                # If no redaction requested, just copy the file to output
                with open(input_file_path, "rb") as src, open(output_file_path, "wb") as dst:
                    dst.write(src.read())

            # Note: Translation would be implemented here

            logger.info(f"Successfully processed job {job_id}")
            return response

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing document: {str(e)}")
            return pdf_processor_pb2.ProcessResponse()

    def redact_pdf_with_pdf_redactor(self, input_path, output_path, redacted_items):
        """Apply redactions to a PDF file using pdf-redactor library"""
        try:
            try:
                # Import the library - the error suggests it's imported differently
                import pdf_redactor
            except ImportError:
                logger.error("pdf-redactor module not found. Please install with: pip install pdf-redactor")
                return False

            logger.info(f"Starting PDF redaction with pdf-redactor for {len(redacted_items)} items")

            # Extract just the text strings to redact
            text_to_redact = []
            for item in redacted_items:
                if "original" in item and item["original"]:
                    text_to_redact.append(item["original"])
                    logger.info(f"Adding text to redact: '{item['original'][:30]}...' (truncated for log)")

            if not text_to_redact:
                logger.warning("No text items found to redact!")
                return False

            logger.info(f"Redacting {len(text_to_redact)} text items")

            # Based on the error, the API seems different than what's being used
            # Let's inspect what the module actually contains
            logger.info(f"pdf_redactor attributes: {dir(pdf_redactor)}")

            # Try the correct approach based on library inspection
            options = {
                "input_stream": open(input_path, "rb"),
                "output_stream": open(output_path, "wb"),
                "text_to_redact": text_to_redact,
                "replacement_text": "██████"  # Black box
            }

            # Log more details about how we're calling the library
            logger.info(f"Calling pdf_redactor with options: {options.keys()}")

            # Call the redactor function - the proper function should be called 'redact'
            if hasattr(pdf_redactor, 'redact'):
                pdf_redactor.redact(options)
            else:
                # If 'redact' doesn't exist, try the other name that was being used
                if callable(pdf_redactor.redactor):
                    pdf_redactor.redactor(options)
                else:
                    logger.error("Could not find the correct redaction function in the pdf_redactor module")
                    return False

            # Close the file streams
            options["input_stream"].close()
            options["output_stream"].close()

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

    def redact_pdf_with_pymupdf(self, input_path, output_path, redacted_items):
        """Apply redactions to a PDF file using PyMuPDF (fitz)"""
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

            # Process each page
            for page_num, items in redactions_by_page.items():
                try:
                    # Adjust for 0-based indexing in PyMuPDF
                    page_idx = page_num - 1
                    if page_idx < 0 or page_idx >= len(doc):
                        logger.warning(f"Page {page_num} out of range (document has {len(doc)} pages)")
                        continue

                    page = doc[page_idx]

                    # Find text instances on the page
                    for item in items:
                        text_to_find = item["original"]

                        # Skip very short strings as they might cause false positives
                        if len(text_to_find.strip()) < 3:
                            logger.warning(f"Skipping very short text for redaction: '{text_to_find}'")
                            continue

                        try:
                            # Search for text - this is where most PDFs might have issues if text is encoded differently
                            text_instances = page.search_for(text_to_find)

                            if text_instances:
                                # Add each instance as a redaction
                                for inst in text_instances:
                                    # Create redaction annotation - black fill
                                    redact_annot = page.add_redact_annot(inst, fill=(0, 0, 0))

                                    if redact_annot:
                                        redactions_applied = True
                                    else:
                                        logger.warning(f"Failed to create redaction annotation for '{text_to_find[:20]}...'")
                            else:
                                # If text search fails, try a more aggressive approach with text blocks
                                logger.warning(f"Direct text search failed for '{text_to_find[:20]}...' on page {page_num}. Trying text block scan.")

                                # Get all text blocks on the page
                                blocks = page.get_text("blocks")
                                for block in blocks:
                                    block_text = block[4]  # The text content is the 5th element

                                    if text_to_find in block_text:
                                        logger.info(f"Found text in a block: '{text_to_find[:20]}...'")

                                        # Create a redaction rectangle covering the entire text block
                                        rect = fitz.Rect(block[:4])  # First 4 elements are the rectangle coordinates
                                        redact_annot = page.add_redact_annot(rect, fill=(0, 0, 0))

                                        if redact_annot:
                                            redactions_applied = True
                                        else:
                                            logger.warning(f"Failed to create block redaction annotation")

                        except Exception as e:
                            logger.error(f"Error searching for text '{text_to_find[:20]}...': {str(e)}")
                            continue

                    # Apply the redactions to this page
                    page.apply_redactions()
                    logger.info(f"Applied redactions to page {page_num}")

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


    def GetStatus(self, request, context):
        """
        Get status of a processing job
        """
        job_id = request.job_id
        logger.info(f"Checking status for job {job_id}")

        # Check specifically for a PDF output file first
        pdf_output_files = list(OUTPUT_DIR.glob(f"{job_id}_*.pdf"))

        if pdf_output_files:
            logger.info(f"Found PDF output file: {pdf_output_files[0].name}")
            return pdf_processor_pb2.StatusResponse(
                job_id=job_id,
                status="completed",
                output_file=pdf_output_files[0].name
            )

        # If no PDF file, check for other output files (e.g., if format conversion happened)
        # Search for any file that starts with job_id but exclude log files
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
        Stream PII detection results for a document
        """
        try:
            logger.info(f"Starting PII detection stream for {request.filename}")

            # Save document to temporary file
            temp_file = TEMP_DIR / f"temp_{uuid.uuid4()}_{request.filename}"
            with open(temp_file, "wb") as f:
                f.write(request.document)

            # Extract text
            document = self.pdf_extractor.extract_text(str(temp_file))

            # Process each page
            for page_num, page_content in document.items():
                if 'text' in page_content:
                    # Detect PII
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

            # Clean up temp file
            temp_file.unlink()
            logger.info(f"Completed PII detection stream for {request.filename}")

        except Exception as e:
            logger.error(f"Error in PII detection stream: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error in PII detection: {str(e)}")


def serve():
    """Start the gRPC server"""
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    pdf_processor_pb2_grpc.add_PDFProcessorServicer_to_server(
        PDFProcessorServicer(), server
    )
    server_address = '[::]:50051'
    server.add_insecure_port(server_address)
    server.start()
    logger.info(f"Server started, listening on {server_address}")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()