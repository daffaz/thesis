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
                 enable_multithreading: bool = True, language: str = "en"):
        """
        Initialize the PII detector with optimized loading and configuration.

        Args:
            model_name: Name of the spaCy model to use
            chunk_size: Default size for text chunks when processing large documents
            enable_multithreading: Whether to use multithreading for large documents
            language: Language code ('en' for English, 'id' for Indonesian)
        """
        # Load model lazily when needed
        self._nlp = None
        self._model_name = model_name
        self.redaction_marker = "***"  # Default redaction marker
        self.chunk_size = chunk_size
        self.enable_multithreading = enable_multithreading
        self.max_workers = min(4, (multiprocessing.cpu_count() or 1) * 2)
        self.language = language

        # Set up regex patterns based on language
        self._setup_regex_patterns()

        # Set up non-PII terms to exclude
        self._setup_non_pii_terms()

        # Other initialization code...
        logger.info(f"Initialized PIIDetector with model '{model_name}' for language '{language}' (lazy loading)")

    def _setup_regex_patterns(self):
        """Set up regex patterns for PII detection based on language"""
        # Common patterns across languages
        self.regex_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'ip_address': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
        }

        if self.language == "id":
            # Indonesian-specific patterns
            self.regex_patterns.update({
                # Enhanced phone pattern for Indonesian formats
                'phone': re.compile(r'(?:\+62[\s\-]?\d{1,3}[\s\-]?\d{3,4}[\s\-]?\d{3,5})'  # Mobile format: +62-812-3456-7890
                                    r'|(?:0\d{2,3}[\s\-]?\d{3,4}[\s\-]?\d{3,4})'  # Mobile format: 0812-3456-7890
                                    r'|(?:\(\d{2,3}\)[\s\-]?\d{3,4}[\s\-]?\d{3,4})'),  # Landline format: (021) 5230-8765

                # NIK pattern (various formats)
                'nik': re.compile(r'\b\d{16}\b|\b\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{4}\b|\b\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\b'),

                # NPWP pattern
                'npwp': re.compile(r'\b\d{2}\.\d{3}\.\d{3}\.\d{1}-\d{3}\.\d{3}\b'),

                # Address pattern for Indonesian addresses
                'address': re.compile(r'\b(?:Jalan|Jl\.?|Kompleks|Komp\.?)\s+[A-Za-z0-9\s]+(?:No\.?|Nomor)?\s*\d+[A-Za-z]?\b,?\s*(?:[A-Za-z\s]+),?\s*\d{5}\b'),

                # Modified credit card to avoid conflict with NIK
                'credit_card': re.compile(r'\b(?:\d{4}[\s\-\.]{1,2}\d{4}[\s\-\.]{1,2}\d{4}[\s\-\.]{1,2}\d{4})\b'),
            })
        else:
            # English patterns
            self.regex_patterns.update({
                'phone': re.compile(r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b'),
                'ssn': re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'),
                'credit_card': re.compile(r'\b(?:\d{4}[-\s.]?){3}\d{4}\b|\b\d{16}\b'),
                'date_of_birth': re.compile(r'\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19|20)\d{2}\b'),
                'address': re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Plaza|Pl)\b,?\s*[A-Za-z\s]+,?\s*[A-Z]{2}\s*\d{5}(-\d{4})?\b')
            })

    def _setup_non_pii_terms(self):
        """Set up non-PII terms to exclude from detection"""
        # Common business terms across languages
        self.non_pii_terms = [
            "company", "corporation", "limited", "inc", "ltd", "llc", "plc",
            "department", "division", "unit", "team", "project"
        ]

        if self.language == "id":
            # Indonesian business and document terms to exclude
            self.non_pii_terms.extend([
                "rp", "rupiah",

                # Document sections
                "proposal", "bisnis", "implementasi", "sistem", "erp", "ringkasan", "eksekutif",
                "latar", "belakang", "perusahaan", "rencana", "implementasi", "proyeksi", "keuangan",
                "analisis", "risiko", "strategi", "mitigasi", "fase", "kontak", "informasi",

                # Business terms
                "direktur", "manajer", "produksi", "manufaktur", "komponen", "elektronik",
                "departemen", "keuangan", "sumber", "daya", "manusia", "manajemen", "rantai",
                "pasokan", "anggaran", "vendor", "konsultan", "bulan", "tahun", "proses",
                "teknologi", "pelatihan", "aplikasi", "module", "dashboard", "analytics",

                # Company prefixes/suffixes
                "pt", "cv", "persero", "tbk"
            ])
        else:
            # English business and document terms to exclude
            self.non_pii_terms.extend([
                "executive", "summary", "introduction", "background", "overview", "proposal",
                "project", "implementation", "plan", "analysis", "risk", "strategy", "phase",
                "contact", "information", "director", "manager", "production", "manufacturing",
                "finance", "human", "resources", "supply", "chain", "budget", "vendor", "consultant"
            ])

    def detect_nik_with_context(self, text: str) -> List[Dict]:
        """
        Specialized detection for Indonesian NIK numbers with context clues.

        Args:
            text: Text to search

        Returns:
            List of detected NIK instances with position information
        """
        if self.language != "id":
            return []

        results = []

        # Patterns for NIK with context labels
        nik_context_patterns = [
            # NIK with label patterns
            r'(?:NIK|KTP|Nomor\s+Induk\s+Kependudukan)(?:\s*:?\s*)(\d{16}|\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2})',
            r'\(NIK:\s*(\d{16}|\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2})\)',
            r'(?:NIK|KTP)(?:\s*[:=]\s*)(\d{16}|\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2})',
        ]

        for pattern in nik_context_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Get the NIK part - if there's a capture group, use it
                if match.lastindex:
                    nik_text = match.group(1)
                    start_pos = match.start(1)
                    end_pos = match.end(1)
                else:
                    nik_text = match.group(0)
                    start_pos = match.start()
                    end_pos = match.end()

                results.append({
                    'text': nik_text,
                    'start': start_pos,
                    'end': end_pos,
                    'type': 'nik',
                    'method': 'context_detection'
                })

        return results

    def _setup_pii_type_mapping(self):
        """Set up PII type mapping based on language"""
        # Common mappings across languages
        self.pii_type_mapping = {
            'email': 'email',
            'credit_card': 'credit_card',
            'ip_address': 'ip_address',
        }

        # Add language-specific mappings
        if self.language == "en":
            self.pii_type_mapping.update({
                'person': 'PERSON',
                'organization': 'ORG',
                'location': 'GPE',
                'address': 'address',
                'nationality': 'NORP',
                'facility': 'FAC',
                'phone': 'phone',
                'ssn': 'ssn',
                'date_of_birth': 'date_of_birth'
            })
        elif self.language == "id":
            self.pii_type_mapping.update({
                'person': 'PERSON',
                'organization': 'ORG',
                'location': 'GPE',
                'address': 'address',
                'phone': 'phone',
                'nik': 'nik',
                'npwp': 'npwp'
            })

    def _setup_context_indicators(self):
        """Set up context indicators for verification based on language"""
        # Name-related context indicators for verification
        if self.language == "en":
            self.name_indicators = [
                "name:", "full name:", "by:", "from:", "to:",
                "sincerely,", "regards,", "signed by", "signature",
                "mr.", "mrs.", "ms.", "dr.", "prof."
            ]

            # Additional title prefixes to improve name detection
            self.title_prefixes = [
                "Mr.", "Mrs.", "Ms.", "Miss", "Dr.", "Prof.", "Sir", "Madam",
                "Lord", "Lady", "Rev.", "Hon."
            ]
        elif self.language == "id":
            self.name_indicators = [
                "nama:", "nama lengkap:", "oleh:", "dari:", "kepada:", "ditujukan kepada:",
                "hormat saya,", "hormat kami,", "salam,", "tertanda,", "tanda tangan",
                "bapak", "ibu", "sdr.", "sdri.", "saudara", "saudari", "dr.", "prof."
            ]

            # Indonesian title prefixes
            self.title_prefixes = [
                "Bapak", "Ibu", "Sdr.", "Sdri.", "Saudara", "Saudari", "Dr.", "Prof.", "Ir.",
                "Haji", "Hajah", "H.", "Hj.", "Drs.", "Drg.", "Jenderal", "Kolonel", "Capt."
            ]

    def detect_indonesian_names(self, text: str) -> List[Dict]:
        """
        Specialized detection for Indonesian names with context.

        Args:
            text: Text to analyze

        Returns:
            List of detected name instances
        """
        if self.language != "id":
            return []

        results = []

        # Common Indonesian name patterns with context
        name_patterns = [
            # People with titles
            r'(?:Bapak|Ibu|Sdr\.|Sdri\.|Dr\.|Ir\.|Prof\.|H\.|Hj\.|Drs\.|Drg\.)[\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',

            # Name with position
            r'(?:Direktur|Manajer|Manager|Kepala|Ketua|Wakil)[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',

            # Name with NIK
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})[\s]*\(NIK:',

            # Name followed by email or contact
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})[\s]*(?:Email|HP|Telp|Telepon):'
        ]

        for pattern in name_patterns:
            for match in re.finditer(pattern, text):
                # Extract the name
                if match.lastindex:
                    name = match.group(1)
                    start_pos = match.start(1)
                    end_pos = match.end(1)
                else:
                    name = match.group(0)
                    start_pos = match.start()
                    end_pos = match.end()

                # Check if this looks like a valid name
                words = name.split()
                if len(words) >= 2 and all(word[0].isupper() for word in words):
                    results.append({
                        'text': name,
                        'start': start_pos,
                        'end': end_pos,
                        'type': 'person',
                        'method': 'indonesian_name_detection'
                    })

        return results

    @property
    def nlp(self):
        """Lazy load the spaCy model when needed based on language"""
        if self._nlp is None:
            if self.language == "id":
                try:
                    # Try to load a multilingual model for Indonesian
                    logger.info(f"Loading multilingual spaCy model for Indonesian")
                    self._nlp = spacy.load("xx_ent_wiki_sm")
                except OSError:
                    # Fall back to English if multilingual model isn't available
                    logger.warning(f"Multilingual model not available, using English model with Indonesian adaptations")
                    self._nlp = spacy.load("en_core_web_md")
                    # Add custom pipeline components to help with Indonesian
                    # self._nlp.add_pipe("custom_id_entity_ruler", last=True)
            else:
                # Default to English
                logger.info(f"Loading spaCy model '{self._model_name}'")
                self._nlp = spacy.load(self._model_name)
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
        logger = logging.getLogger(__name__)
        logger.info(f"Starting regex detection for types: {regex_types_to_include}")

        # Check if this is a large text that should be chunked
        if len(text) > self.chunk_size:
            return self.detect_pii_in_chunks(text, pii_types)
            
        # Process the text
        pii_instances = self._process_text(text, ner_types_to_include, regex_types_to_include)
        
        # Special handling for credit cards - try to detect fragments if no credit cards found
        if 'credit_card' in regex_types_to_include:
            credit_card_found = any(item['type'] == 'credit_card' for item in pii_instances)
            if not credit_card_found:
                # Try to detect and reassemble credit card fragments
                credit_card_fragments = self.detect_credit_card_fragments(text)
                if credit_card_fragments:
                    logger.info(f"Successfully reassembled {len(credit_card_fragments)} credit cards from fragments")
                    pii_instances.extend(credit_card_fragments)
        
        return pii_instances
        
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
                        logger.info(f"Searching for {pii_type} with pattern: {pattern}")
                        matches = list(re.finditer(pattern, text))
                        logger.info(f"Found {len(matches)} matches for {pii_type}")
                        
                        if pii_type == 'credit_card' and not matches:
                            # Look for any digit sequences that might be credit cards
                            digit_seqs = re.findall(r'\d{4,}', text)
                            if digit_seqs:
                                logger.info(f"Found digit sequences that might be parts of CC: {digit_seqs[:10]}")

                        for match in pattern.finditer(text):
                            found_text = match.group()
                            logger.info(f"Found {pii_type}: {found_text[:4]}...{found_text[-4:]} at position {match.start()}-{match.end()}")
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
        Apply additional verification for PERSON entities to reduce false positives.

        Args:
            entity: spaCy entity to verify
            text: Full text for context

        Returns:
            Boolean indicating if this should be treated as PII
        """
        entity_text = entity.text.strip()

        # Skip very short entities - likely false positives
        if len(entity_text) < 4:
            return False

        # Skip entities that are in our non-PII list
        if entity_text.lower() in self.non_pii_terms:
            return False

        # Skip words that are document section headings
        if self.language == "id":
            indonesian_headers = ["proposal", "bisnis", "implementasi", "sistem", "ringkasan",
                                  "eksekutif", "perusahaan", "proyeksi", "keuangan", "produksi"]
            for header in indonesian_headers:
                if entity_text.lower() == header:
                    return False

        # Skip all-uppercase words that are likely not names (but acronyms, headers, etc.)
        if entity_text.isupper() and len(entity_text) > 3:
            return False

        # Check capitalization - names should be capitalized
        if not entity_text[0].isupper():
            return False

        # Get surrounding context
        context_start = max(0, entity.start_char - 50)
        context_end = min(len(text), entity.end_char + 50)
        surrounding = text[context_start:context_end].lower()

        # Look for name indicators in context (language-specific)
        if self.language == "id":
            name_indicators = [
                "nama:", "bapak", "ibu", "sdr.", "sdri.", "tuan", "nyonya", "dr.", "ir.",
                "direktur", "manager", "oleh:", "dengan", "kepada:", "ditujukan", "hormat",
                "tertanda"
            ]
        else:
            name_indicators = [
                "name:", "mr.", "mrs.", "ms.", "dr.", "prof.", "by:", "from:", "to:",
                "sincerely,", "regards,", "signed"
            ]

        for indicator in name_indicators:
            if indicator in surrounding:
                return True

        # Special case for Indonesian names with multiple capitalized words
        if self.language == "id" and len(entity_text.split()) > 1:
            # If all words are capitalized, it's more likely a name
            if all(word[0].isupper() for word in entity_text.split() if word):
                # Make sure not all words are in the non-PII terms
                if not all(word.lower() in self.non_pii_terms for word in entity_text.split()):
                    return True

        # If no evidence this is a name, assume it's not PII
        return False

    def detect_indonesian_nik(self, text: str) -> List[Dict]:
        """
        Specialized function to detect Indonesian NIK numbers with context.

        Args:
            text: Text to analyze

        Returns:
            List of detected NIK instances
        """
        if self.language != "id":
            return []

        results = []

        # Patterns to detect NIK with context
        nik_patterns = [
            r'(?:NIK|KTP|Nomor\s+Induk\s+Kependudukan|Nomor\s+KTP)(?:\s*:?\s*)(\d{16})',
            r'(?:NIK|KTP|Nomor\s+Induk\s+Kependudukan|Nomor\s+KTP)(?:\s*:?\s*)(\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{4})',
            r'(?:NIK|KTP|Nomor\s+Induk\s+Kependudukan|Nomor\s+KTP)(?:\s*:?\s*)(\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2})',
            r'\(NIK:\s*(\d{16})\)',
            r'\(NIK:\s*(\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2})\)'
        ]

        for pattern in nik_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                nik_text = match.group(1) if match.lastindex else match.group(0)
                results.append({
                    'text': nik_text,
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'nik',
                    'method': 'nik_detector'
                })

        return results

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
        pii_by_type = {}
        for item in pii_instances:
            pii_type = item['type']
            pii_by_type[pii_type] = pii_by_type.get(pii_type, 0) + 1    
            key = (item.get('text', ''), item.get('type', ''))
            if key not in seen:
                seen.add(key)
                unique_instances.append(item)
        
        logger.info(f"PII types found: {pii_by_type}")
                
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

        logger = logging.getLogger(__name__)
        logger.info(f"Detected {len(pii_instances)} PII instances to redact")

        # Make a copy of the original text
        redacted_text = text

        # Apply redactions from end to beginning to avoid offset issues
        redacted_items = []
        for item in sorted(pii_instances, key=lambda x: x['start'], reverse=True):
            # TODO REMOVE THIS LATER
            logger.info(f"Redacting {item['type']}: {item['text'][:4]}...{item['text'][-4:]} at position {item['start']}-{item['end']}")
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
        
    def detect_credit_card_fragments(self, text: str) -> List[Dict]:
        """
        Detect credit card fragments and attempt to reassemble them
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected credit card instances
        """
        logger = logging.getLogger(__name__)
        results = []
        
        # Look for sequences of 4 digits with optional separators
        # The pattern matches both fragments and complete card numbers
        fragments = re.findall(r'\b\d{4}\b', text)
        complete_cards = re.findall(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', text)
        
        # Log any complete cards found directly
        if complete_cards:
            logger.info(f"Found {len(complete_cards)} complete credit cards directly: {complete_cards}")
            for card in complete_cards:
                pos = text.find(card)
                if pos >= 0:
                    results.append({
                        'text': card,
                        'start': pos,
                        'end': pos + len(card),
                        'type': 'credit_card',
                        'method': 'direct_match'
                    })
        
        # Now try to reassemble fragments
        logger.info(f"Found {len(fragments)} potential credit card fragments: {fragments}")
        
        # Create sliding window of 4 fragments
        for i in range(len(fragments) - 3):
            # Look for 4 consecutive fragments that could form a card
            potential_fragments = fragments[i:i+4]
            
            # Check if these fragments look sequential
            reassembled = '-'.join(potential_fragments)
            
            # Log potential reassembly
            logger.info(f"Potential reassembled credit card: {reassembled}")
            
            # Don't validate too strictly - if it has 16 digits, treat it as a potential card
            digits_only = ''.join(potential_fragments)
            if len(digits_only) == 16:
                pos = text.find(potential_fragments[0])
                if pos >= 0:
                    # Get an approximation of where the last fragment ends
                    last_pos = text.find(potential_fragments[3], pos)
                    if last_pos >= 0:
                        end_pos = last_pos + len(potential_fragments[3])
                        
                        results.append({
                            'text': reassembled,
                            'start': pos,
                            'end': end_pos,
                            'type': 'credit_card',
                            'method': 'fragment_reassembly'
                        })
                        logger.info(f"Successfully reassembled credit card: {reassembled}")
        
        # Also check for number sequences that look like credit cards without separators
        # This catches cases where the PDF extraction removes the separators
        unseparated = re.findall(r'\b\d{16}\b', text)
        if unseparated:
            logger.info(f"Found {len(unseparated)} unseparated 16-digit numbers: {unseparated}")
            for card in unseparated:
                pos = text.find(card)
                if pos >= 0:
                    # Format with dashes for better readability
                    formatted = f"{card[0:4]}-{card[4:8]}-{card[8:12]}-{card[12:16]}"
                    results.append({
                        'text': formatted,
                        'start': pos,
                        'end': pos + len(card),
                        'type': 'credit_card',
                        'method': 'unseparated'
                    })
                    logger.info(f"Detected unseparated credit card: {card} -> {formatted}")
        
        return results

    def _is_valid_credit_card(self, card_number: str) -> bool:
        """
        Basic validation for a credit card number
        
        Args:
            card_number: Credit card number with or without separators
            
        Returns:
            Boolean indicating if it looks like a valid card
        """
        # Remove separators
        digits = re.sub(r'[^0-9]', '', card_number)
        
        # Check length
        if len(digits) != 16:
            return False
        
        # Check starting digit (simple check)
        valid_starts = ['4', '5', '3', '6']  # Visa, MC, Amex, Discover
        if digits[0] not in valid_starts:
            return False
        
        return True
    
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