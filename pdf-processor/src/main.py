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
            logger.info(f"Extracted text: {document}")

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
                redaction_types = list(request.options.redaction_types) if request.options.redaction_types else None
                redacted_document, redacted_items = self.pii_detector.redact_document(
                    document, redaction_types
                )

                # Get redaction statistics
                stats = self.pii_detector.get_pii_statistics(document)

                # Create RedactionStats message
                redaction_stats = pdf_processor_pb2.RedactionStats(
                    total_pii_count=stats['total_pii_count']
                )

                # Convert dict to protobuf map fields
                for key, value in stats['by_type'].items():
                    redaction_stats.by_type[key] = value

                for key, value in stats['by_method'].items():
                    redaction_stats.by_method[key] = value

                for key, value in stats['by_page'].items():
                    # Convert page number to string for protobuf map
                    redaction_stats.by_page[str(key)] = value

                response.redaction_stats.CopyFrom(redaction_stats)

                # TODO: In a real implementation, you would reconstruct the PDF with redactions
                # For now, just copy the file to demonstrate the flow
                with open(input_file_path, "rb") as src, open(output_file_path, "wb") as dst:
                    dst.write(src.read())

            # Note: Translation would be implemented here

            logger.info(f"Successfully processed job {job_id}")
            return response

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing document: {str(e)}")
            return pdf_processor_pb2.ProcessResponse()

    def GetStatus(self, request, context):
        """
        Get status of a processing job
        """
        job_id = request.job_id
        logger.info(f"Checking status for job {job_id}")

        # Check if output file exists
        output_files = list(OUTPUT_DIR.glob(f"{job_id}_*"))

        if output_files:
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