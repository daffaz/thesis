import re
import spacy
from typing import Dict, List, Set, Tuple, Optional, Union
import concurrent.futures
from functools import lru_cache
import logging
import multiprocessing

# Set up logging
logger = logging.getLogger(__name__)

class PIIDetector:
    """
    Detects and redacts personally identifiable information (PII) from text
    using a combination of NER models and regex patterns with optimized performance.
    """
    
    def __init__(self, model_name: str = "en_core_web_md", chunk_size: int = 5000, 
                 enable_multithreading: bool = True):
        """
        Initialize the PII detector with optimized loading and configuration.
        
        Args:
            model_name: Name of the spaCy model to use
            chunk_size: Default size for text chunks when processing large documents
            enable_multithreading: Whether to use multithreading for large documents
        """
        # Load model lazily when needed
        self._nlp = None
        self._model_name = model_name
        self.redaction_marker = "***"  # Default redaction marker
        self.chunk_size = chunk_size
        self.enable_multithreading = enable_multithreading
        self.max_workers = min(4, ( multiprocessing.cpu_count() or 1) * 2)
        
        # Set up regex patterns for common PII types not well-covered by NER
        # Precompile regex patterns for better performance
        self.regex_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'),
            'credit_card': re.compile( r'(?:\d{4}[-\s]?){3}\d{4}|(?:\d{4}[\s-]?){3}\d{4}'),
            'ip_address': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
            'date_of_birth': re.compile(r'\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19|20)\d{2}\b')
        }

        # Define mapping between user-friendly PII type names and internal representations
        self.pii_type_mapping = {
            'person': 'PERSON',
            # 'organization': 'ORG', # will turn this on in future
            'location': 'GPE',
            'address': 'LOC',
            'nationality': 'NORP',
            'facility': 'FAC',
            'email': 'email',
            'phone': 'phone',
            'ssn': 'ssn',
            'credit_card': 'credit_card',
            'ip_address': 'ip_address',
            'date_of_birth': 'date_of_birth'
        }

        # Name-related context indicators for verification
        self.name_indicators = [
            "name:", "nama:", "full name:", "by:", "from:", "to:",
            "sincerely,", "regards,", "signed by", "signature"
        ]
        
        # Additional title prefixes to improve name detection
        self.title_prefixes = [
            "Mr.", "Mrs.", "Ms.", "Miss", "Dr.", "Prof.", "Sir", "Madam", 
            "Lord", "Lady", "Rev.", "Hon."
        ]
        
        logger.info(f"Initialized PIIDetector with model '{model_name}' (lazy loading)")

    @property
    def nlp(self):
        """Lazy load the spaCy model when needed"""
        if self._nlp is None:
            logger.info(f"Loading spaCy model '{self._model_name}'")
            self._nlp = spacy.load(self._model_name)
            # Add custom pipeline components if needed
            # self._nlp.add_pipe("custom_component", last=True)
        return self._nlp

    def set_redaction_marker(self, marker: str) -> None:
        """
        Set the marker used for redacting PII.

        Args:
            marker: String to replace PII with
        """
        self.redaction_marker = marker

    def detect_pii(self, text: str, pii_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Detect PII in the given text.

        Args:
            text: Text to analyze for PII
            pii_types: List of PII types to detect (if None, detect all)

        Returns:
            List of dictionaries with information about detected PII
        """
        if not text or not text.strip():
            return []

        # Determine which PII types to detect
        ner_types_to_include, regex_types_to_include = self._resolve_pii_types(pii_types)
        
        # Check if this is a large text that should be chunked
        if len(text) > self.chunk_size:
            return self.detect_pii_in_chunks(text, pii_types)
            
        # Process the text
        return self._process_text(text, ner_types_to_include, regex_types_to_include)
        
    def _resolve_pii_types(self, pii_types: Optional[List[str]]) -> Tuple[List[str], List[str]]:
        """
        Resolve the requested PII types to internal representation.
        
        Args:
            pii_types: User-specified PII types or None for all
            
        Returns:
            Tuple of (ner_types_to_include, regex_types_to_include)
        """
        # Default values - detect everything
        ner_types_to_include = ['PERSON', 'GPE', 'LOC', 'NORP', 'FAC']
        regex_types_to_include = list(self.regex_patterns.keys())

        # Filter based on provided pii_types
        if pii_types is not None:
            # Convert user-friendly types to internal types
            normalized_types = []
            for pii_type in pii_types:
                pii_type_lower = pii_type.lower()
                if pii_type_lower in self.pii_type_mapping:
                    normalized_types.append(self.pii_type_mapping[pii_type_lower])
                else:
                    # If it's already in internal format
                    normalized_types.append(pii_type)

            # Filter NER types and regex types based on normalized types
            ner_types_to_include = [t for t in ner_types_to_include if t in normalized_types]
            regex_types_to_include = [t for t in regex_types_to_include if t in normalized_types]
            
        return ner_types_to_include, regex_types_to_include
    
    def _process_text(self, text: str, ner_types_to_include: List[str], 
                     regex_types_to_include: List[str]) -> List[Dict]:
        """
        Process a single chunk of text to detect PII.
        
        Args:
            text: Text to analyze
            ner_types_to_include: NER entity types to detect
            regex_types_to_include: Regex pattern types to use
            
        Returns:
            List of detected PII instances
        """
        pii_instances = []

        try:
            # Use spaCy NER for named entities if any NER types are included
            if ner_types_to_include:
                doc = self.nlp(text)

                # Extract entities detected by spaCy
                for ent in doc.ents:
                    if ent.label_ in ner_types_to_include:
                        # For PERSON entities, apply verification
                        if ent.label_ == 'PERSON' and not self._verify_person_entity(ent, text):
                            continue

                        pii_instances.append({
                            'text': ent.text,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'type': ent.label_,
                            'method': 'ner'
                        })

            # Use regex for additional PII types
            if regex_types_to_include:
                for pii_type, pattern in self.regex_patterns.items():
                    if pii_type in regex_types_to_include:
                        for match in pattern.finditer(text):
                            pii_instances.append({
                                'text': match.group(),
                                'start': match.start(),
                                'end': match.end(),
                                'type': pii_type,
                                'method': 'regex'
                            })

            # Sort by starting position
            pii_instances.sort(key=lambda x: x['start'])

            # Remove duplicates
            return self._remove_duplicates(pii_instances)

        except Exception as e:
            logger.error(f"Error in detect_pii: {str(e)}")
            return []
            
    def detect_pii_in_chunks(self, text: str, pii_types: Optional[List[str]] = None, 
                         overlap: int = 200) -> List[Dict]:
        """
        Detect PII in a large text by processing it in overlapping chunks.
        Uses multithreading if enabled.
        
        Args:
            text: Text to analyze for PII
            pii_types: List of PII types to detect (if None, detect all)
            overlap: Overlap between chunks to avoid missing PII that spans chunk boundaries
            
        Returns:
            List of dictionaries with information about detected PII
        """
        if not text or not text.strip():
            return []
        
        # If the text is small enough, process it directly
        if len(text) <= self.chunk_size:
            return self.detect_pii(text, pii_types)
        
        # Resolve PII types once for all chunks
        ner_types, regex_types = self._resolve_pii_types(pii_types)
        
        # Create chunks
        chunks = []
        offset = 0
        
        while offset < len(text):
            # Get the chunk with overlap
            end = min(offset + self.chunk_size, len(text))
            chunks.append((text[offset:end], offset))
            
            # Move to the next chunk with overlap
            offset = end - overlap if end < len(text) else len(text)
            
        # Process chunks
        all_pii = []
        
        if self.enable_multithreading and len(chunks) > 1:
            # Process chunks in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for chunk_text, chunk_offset in chunks:
                    future = executor.submit(
                        self._process_chunk, chunk_text, chunk_offset, ner_types, regex_types
                    )
                    futures.append(future)
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    all_pii.extend(future.result())
        else:
            # Process chunks sequentially
            for chunk_text, chunk_offset in chunks:
                chunk_pii = self._process_chunk(chunk_text, chunk_offset, ner_types, regex_types)
                all_pii.extend(chunk_pii)
        
        # Remove duplicates that might occur in overlapping regions
        return self._remove_duplicates(all_pii)
    
    def _process_chunk(self, chunk_text: str, offset: int, ner_types: List[str], 
                      regex_types: List[str]) -> List[Dict]:
        """
        Process a single chunk and adjust the positions.
        
        Args:
            chunk_text: Text chunk to process
            offset: Position of this chunk in the original text
            ner_types: NER types to include
            regex_types: Regex types to include
            
        Returns:
            List of PII instances with positions adjusted for the original text
        """
        # Process the chunk
        chunk_pii = self._process_text(chunk_text, ner_types, regex_types)
        
        # Adjust positions
        for item in chunk_pii:
            item['start'] += offset
            item['end'] += offset
            
        return chunk_pii

    def _verify_person_entity(self, entity, text: str) -> bool:
        """
        Enhanced verification for PERSON entities to reduce false positives.

        Args:
            entity: spaCy entity to verify
            text: Full text for context

        Returns:
            Boolean indicating if this should be treated as PII
        """
        entity_text = entity.text.strip()

        # Skip very short entities - likely false positives
        if len(entity_text) < 3:
            return False
            
        # Skip single-word lowercase entities - likely not names
        if len(entity_text.split()) == 1 and entity_text[0].islower():
            return False

        # Get surrounding context
        context_start = max(0, entity.start_char - 50)
        context_end = min(len(text), entity.end_char + 50)
        surrounding = text[context_start:context_end].lower()

        # Check for name indicators in context
        for indicator in self.name_indicators:
            if indicator in surrounding:
                return True

        # Check capitalization pattern - names are typically capitalized
        words = entity_text.split()
        if len(words) >= 2 and all(word[0].isupper() for word in words if word):
            return True
            
        # Check for title prefixes
        for title in self.title_prefixes:
            prefix_with_space = title + " "
            if prefix_with_space + entity_text in text or prefix_with_space + entity_text.lower() in text.lower():
                return True

        # Check token part-of-speech tags - names are typically proper nouns
        if all(token.pos_ == "PROPN" for token in entity):
            return True
            
        # Check if entity is part of a larger name pattern
        # Example: "John and Mary Smith" - both should be detected as names
        is_name_pattern = False
        for token in entity.doc:
            if token.text.lower() in ["and", "&"] and token.i > 0 and token.i < len(entity.doc) - 1:
                prev_token = entity.doc[token.i - 1]
                next_token = entity.doc[token.i + 1]
                if (prev_token.ent_type_ == "PERSON" or next_token.ent_type_ == "PERSON"):
                    is_name_pattern = True
                    break
                    
        if is_name_pattern:
            return True

        # If no strong evidence it's a name, assume it's not PII
        return False

    def _remove_duplicates(self, pii_instances: List[Dict]) -> List[Dict]:
        """
        Remove duplicate PII detections using an efficient algorithm.

        Args:
            pii_instances: List of detected PII instances

        Returns:
            Deduplicated list of PII instances
        """
        if not pii_instances:
            return []
            
        # Sort by start position for overlap detection
        pii_instances.sort(key=lambda x: (x['start'], -x['end']))
        
        # Filter overlapping entities, preferring longer matches
        unique_instances = []
        seen = set()  # Track combinations of text and type
        
        # First pass: remove exact text+type duplicates
        for item in pii_instances:
            key = (item.get('text', ''), item.get('type', ''))
            if key not in seen:
                seen.add(key)
                unique_instances.append(item)
                
        # Second pass: remove overlapping entities
        filtered_instances = []
        
        for i, current in enumerate(unique_instances):
            # Check if this instance is overlapped by a longer, higher priority one
            is_overlapped = False
            
            for other in unique_instances:
                # Skip same instance
                if current is other:
                    continue
                    
                # Check for overlap with higher priority entity
                if (other['start'] <= current['start'] and other['end'] >= current['end'] and
                    (other['end'] - other['start'] > current['end'] - current['start'] or
                     (other['method'] == 'ner' and current['method'] == 'regex'))):
                    is_overlapped = True
                    break
                    
            if not is_overlapped:
                filtered_instances.append(current)

        return filtered_instances

    def redact_pii(self, text: str, pii_types: Optional[List[str]] = None) -> Tuple[str, List[Dict]]:
        """
        Redact PII in the given text.

        Args:
            text: Text to redact
            pii_types: List of PII types to redact (if None, redact all)

        Returns:
            Tuple of (redacted text, list of redacted items)
        """
        if not text or not text.strip():
            return text, []

        # Detect PII with the specified types
        pii_instances = self.detect_pii(text, pii_types)

        # Make a copy of the original text
        redacted_text = text

        # Apply redactions from end to beginning to avoid offset issues
        redacted_items = []
        for item in sorted(pii_instances, key=lambda x: x['start'], reverse=True):
            # Get the length of the PII text
            pii_length = item['end'] - item['start']

            # Create a redaction marker of the same length as the original text
            # This helps maintain document formatting
            redaction = self.redaction_marker * (pii_length // len(self.redaction_marker) + 1)
            redaction = redaction[:pii_length]

            # Apply the redaction
            redacted_text = (
                redacted_text[:item['start']] +
                redaction +
                redacted_text[item['end']:]
            )

            redacted_items.append({
                'original': item['text'],
                'redacted': redaction,
                'type': item['type'],
                'position': (item['start'], item['end'])
            })

        return redacted_text, redacted_items

    def redact_document(self, document: Dict[int, Dict],
                        pii_types: Optional[List[str]] = None) -> Tuple[Dict[int, Dict], List[Dict]]:
        """
        Redact PII from an entire document.

        Args:
            document: Document dictionary (from PDFExtractor)
            pii_types: List of PII types to redact (if None, redact all)

        Returns:
            Tuple of (redacted document, list of all redacted items)
        """
        redacted_document = {}
        all_redacted_items = []

        # Process pages in parallel if multithreading is enabled and document is large
        if self.enable_multithreading and len(document) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                
                for page_num, page_content in document.items():
                    futures[executor.submit(self._redact_page, page_content, page_num, pii_types)] = page_num
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    page_num = futures[future]
                    try:
                        redacted_page, redacted_items = future.result()
                        redacted_document[page_num] = redacted_page
                        all_redacted_items.extend(redacted_items)
                    except Exception as e:
                        logger.error(f"Error redacting page {page_num}: {str(e)}")
                        # Copy original page content on error
                        redacted_document[page_num] = document[page_num].copy()
        else:
            # Process pages sequentially
            for page_num, page_content in document.items():
                try:
                    redacted_page, redacted_items = self._redact_page(page_content, page_num, pii_types)
                    redacted_document[page_num] = redacted_page
                    all_redacted_items.extend(redacted_items)
                except Exception as e:
                    logger.error(f"Error redacting page {page_num}: {str(e)}")
                    # Copy original page content on error
                    redacted_document[page_num] = page_content.copy()

        # Deduplicate redacted items
        unique_redacted_items = []
        seen = set()

        for item in all_redacted_items:
            # Create a unique identifier for this item
            if 'original' in item and 'type' in item:
                key = (item['original'], item['type'])
                if key not in seen:
                    seen.add(key)
                    unique_redacted_items.append(item)

        return redacted_document, unique_redacted_items
        
    def _redact_page(self, page_content: Dict, page_num: int, pii_types: Optional[List[str]]) -> Tuple[Dict, List[Dict]]:
        """
        Redact PII from a single page.
        
        Args:
            page_content: Page content dictionary
            page_num: Page number
            pii_types: PII types to redact
            
        Returns:
            Tuple of (redacted page content, redacted items)
        """
        # Make a deep copy of the page content
        redacted_page = page_content.copy()
        page_redacted_items = []

        # Redact main text
        if 'text' in page_content and page_content['text']:
            redacted_text, redacted_items = self.redact_pii(page_content['text'], pii_types)
            redacted_page['text'] = redacted_text

            # Add page information to redacted items
            for item in redacted_items:
                item['page'] = page_num

            page_redacted_items.extend(redacted_items)

        # Redact paragraphs if available
        if 'paragraphs' in page_content and page_content['paragraphs']:
            redacted_paragraphs = []
            for i, paragraph in enumerate(page_content['paragraphs']):
                if not paragraph.strip():
                    redacted_paragraphs.append(paragraph)
                    continue
                    
                redacted_para, para_items = self.redact_pii(paragraph, pii_types)
                redacted_paragraphs.append(redacted_para)

                # Add paragraph and page information
                for item in para_items:
                    item['page'] = page_num
                    item['paragraph'] = i

                page_redacted_items.extend(para_items)

            redacted_page['paragraphs'] = redacted_paragraphs

        # Redact tables if available
        if 'tables' in page_content and page_content['tables']:
            redacted_tables = []
            for i, table in enumerate(page_content['tables']):
                if 'text' in table and table['text'].strip():
                    redacted_table = table.copy()
                    redacted_table_text, table_items = self.redact_pii(table['text'], pii_types)
                    redacted_table['text'] = redacted_table_text
                    redacted_tables.append(redacted_table)

                    # Add table and page information
                    for item in table_items:
                        item['page'] = page_num
                        item['table'] = i

                    page_redacted_items.extend(table_items)
                else:
                    redacted_tables.append(table)

            redacted_page['tables'] = redacted_tables

        return redacted_page, page_redacted_items

    def get_pii_statistics(self, document_id: str, document: Dict[int, Dict],
                          pii_types: Optional[List[str]] = None) -> Dict:
        """
        Get statistics about PII in the document with caching for performance.

        Args:
            document_id: Unique identifier for the document (for caching)
            document: Document dictionary (from PDFExtractor)
            pii_types: List of PII types to include in statistics (if None, include all)

        Returns:
            Dictionary with PII statistics
        """
        all_pii = []

        # Use multithreading for large documents
        if self.enable_multithreading and len(document) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                
                for page_num, page_content in document.items():
                    if 'text' in page_content and page_content['text']:
                        futures[executor.submit(self.detect_pii, page_content['text'], pii_types)] = page_num
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    page_num = futures[future]
                    try:
                        page_pii = future.result()
                        for item in page_pii:
                            item['page'] = page_num
                        all_pii.extend(page_pii)
                    except Exception as e:
                        logger.error(f"Error getting PII statistics for page {page_num}: {str(e)}")
        else:
            # Process pages sequentially
            for page_num, page_content in document.items():
                if 'text' in page_content and page_content['text']:
                    try:
                        page_pii = self.detect_pii(page_content['text'], pii_types)
                        for item in page_pii:
                            item['page'] = page_num
                        all_pii.extend(page_pii)
                    except Exception as e:
                        logger.error(f"Error getting PII statistics for page {page_num}: {str(e)}")

        # Deduplicate PII instances
        unique_pii = []
        seen = set()

        for item in all_pii:
            key = (item.get('text', ''), item.get('type', ''))
            if key not in seen:
                seen.add(key)
                unique_pii.append(item)

        # Calculate statistics using deduplicated PII
        stats = {
            'total_pii_count': len(unique_pii),
            'by_type': {},
            'by_method': {},
            'by_page': {}
        }
        
        # Count by type
        for item in unique_pii:
            # By PII type
            pii_type = item['type']
            if pii_type not in stats['by_type']:
                stats['by_type'][pii_type] = 0
            stats['by_type'][pii_type] += 1
            
            # By detection method
            method = item['method']
            if method not in stats['by_method']:
                stats['by_method'][method] = 0
            stats['by_method'][method] += 1
            
            # By page
            page = item['page']
            page_key = str(page)  # Convert to string for protobuf compatibility
            if page_key not in stats['by_page']:
                stats['by_page'][page_key] = 0
            stats['by_page'][page_key] += 1
        
        return stats