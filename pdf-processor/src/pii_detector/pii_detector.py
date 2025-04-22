import re
import spacy
from typing import Dict, List, Set, Tuple, Optional, Union

class PIIDetector:
    """
    Detects and redacts personally identifiable information (PII) from text
    using a combination of NER models and regex patterns.
    """
    
    def __init__(self, model_name: str = "en_core_web_md"):
        """
        Initialize the PII detector with a spaCy model.
        
        Args:
            model_name: Name of the spaCy model to use
        """
        self.nlp = spacy.load(model_name)
        self.redaction_marker = "***"  # Default redaction marker
        
        # Set up regex patterns for common PII types not well-covered by NER
        self.regex_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b',
            'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[- ]?){3}\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'date_of_birth': r'\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19|20)\d{2}\b'
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
        ner_types_to_include = ['PERSON', 'GPE', 'LOC', 'NORP', 'FAC']
        regex_types_to_include = list(self.regex_patterns.keys())

        # Filter based on provided pii_types
        if pii_types is not None:
            # Convert user-friendly types to internal types
            normalized_types = []
            for pii_type in pii_types:
                if pii_type.lower() in self.pii_type_mapping:
                    normalized_types.append(self.pii_type_mapping[pii_type.lower()])
                else:
                    # If it's already in internal format
                    normalized_types.append(pii_type)

            # Filter NER types and regex types based on normalized types
            ner_types_to_include = [t for t in ner_types_to_include if t in normalized_types]
            regex_types_to_include = [t for t in regex_types_to_include if t in normalized_types]

        pii_instances = []

        try:
            # Use spaCy NER for named entities
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
            for pii_type, pattern in self.regex_patterns.items():
                if pii_type in regex_types_to_include:
                    for match in re.finditer(pattern, text):
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
            print(f"Error in detect_pii: {str(e)}")
            return []

    def _verify_person_entity(self, entity, text: str) -> bool:
        """
        Apply additional verification for PERSON entities to reduce false positives.

        Args:
            entity: spaCy entity to verify
            text: Full text for context

        Returns:
            Boolean indicating if this should be treated as PII
        """
        entity_text = entity.text

        # Skip very short entities - likely false positives
        if len(entity_text.strip()) < 3:
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

        # If no strong evidence it's a name, assume it's not PII
        return False

    def _remove_duplicates(self, pii_instances: List[Dict]) -> List[Dict]:
        """
        Remove duplicate PII detections.

        Args:
            pii_instances: List of detected PII instances

        Returns:
            Deduplicated list of PII instances
        """
        unique_instances = []
        seen = set()  # Track combinations of text and type

        for item in pii_instances:
            key = (item['text'], item['type'])
            if key not in seen:
                seen.add(key)
                unique_instances.append(item)

        return unique_instances

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

        for page_num, page_content in document.items():
            # Make a deep copy of the page content
            redacted_page = page_content.copy()

            # Redact main text
            if 'text' in page_content:
                redacted_text, redacted_items = self.redact_pii(page_content['text'], pii_types)
                redacted_page['text'] = redacted_text

                # Add page information to redacted items
                for item in redacted_items:
                    item['page'] = page_num

                all_redacted_items.extend(redacted_items)

            # Redact paragraphs if available and not already processed through main text
            if 'paragraphs' in page_content:
                redacted_paragraphs = []
                for i, paragraph in enumerate(page_content['paragraphs']):
                    redacted_para, para_items = self.redact_pii(paragraph, pii_types)
                    redacted_paragraphs.append(redacted_para)

                    # Add paragraph and page information
                    for item in para_items:
                        item['page'] = page_num
                        item['paragraph'] = i

                    all_redacted_items.extend(para_items)

                redacted_page['paragraphs'] = redacted_paragraphs

            # Redact tables if available
            if 'tables' in page_content and page_content['tables']:
                redacted_tables = []
                for i, table in enumerate(page_content['tables']):
                    if 'text' in table:
                        redacted_table = table.copy()
                        redacted_table_text, table_items = self.redact_pii(table['text'], pii_types)
                        redacted_table['text'] = redacted_table_text
                        redacted_tables.append(redacted_table)

                        # Add table and page information
                        for item in table_items:
                            item['page'] = page_num
                            item['table'] = i

                        all_redacted_items.extend(table_items)
                    else:
                        redacted_tables.append(table)

                redacted_page['tables'] = redacted_tables

            redacted_document[page_num] = redacted_page

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

    def get_pii_statistics(self, document: Dict[int, Dict],
                          pii_types: Optional[List[str]] = None) -> Dict:
        """
        Get statistics about PII in the document.

        Args:
            document: Document dictionary (from PDFExtractor)
            pii_types: List of PII types to include in statistics (if None, include all)

        Returns:
            Dictionary with PII statistics
        """
        all_pii = []

        for page_num, page_content in document.items():
            # Process main text
            if 'text' in page_content:
                page_pii = self.detect_pii(page_content['text'], pii_types)
                for item in page_pii:
                    item['page'] = page_num
                all_pii.extend(page_pii)

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
            if page not in stats['by_page']:
                stats['by_page'][page] = 0
            stats['by_page'][page] += 1
        
        return stats