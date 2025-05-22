# translation_service.py
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import logging
import hashlib
import json
from datetime import datetime, timedelta
from functools import lru_cache

# Import the NLLB model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, utils as tf_utils
import torch
import langdetect

# Configure logging
logger = logging.getLogger(__name__)

class TranslationCache:
    """Handles caching of translations to improve performance"""
    
    def __init__(self, cache_dir: Optional[str] = None, max_age_days: int = 30):
        self.cache_dir = Path(cache_dir or "/app/data/translation_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(days=max_age_days)
        
    def _get_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate a unique cache key for the translation"""
        content = f"{text}:{source_lang}:{target_lang}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def get(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Retrieve a cached translation if available and not expired"""
        cache_key = self._get_cache_key(text, source_lang, target_lang)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                # Check if cache is expired
                cached_time = datetime.fromisoformat(cache_data['timestamp'])
                if datetime.now() - cached_time > self.max_age:
                    cache_file.unlink()  # Remove expired cache
                    return None
                    
                return cache_data['translation']
            except Exception as e:
                logger.warning(f"Cache read error: {str(e)}")
                return None
        return None
        
    def set(self, text: str, source_lang: str, target_lang: str, translation: str):
        """Store a translation in the cache"""
        cache_key = self._get_cache_key(text, source_lang, target_lang)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            cache_data = {
                'text': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'translation': translation,
                'timestamp': datetime.now().isoformat()
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Cache write error: {str(e)}")

class TranslationService:
    """
    Handles translation of text between languages using the NLLB model.
    Supports only English ↔ Indonesian translation.
    """

    def __init__(self, model_path: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize the translation service with an NLLB model

        Args:
            model_path: Path to a local NLLB model, or None to download from HF
            cache_dir: Directory to store translation cache
        """
        # Define supported language pairs
        self.SUPPORTED_LANGUAGES = {
            "eng_Latn": "English",
            "ind_Latn": "Indonesian"
        }
        
        # Model configuration
        self.model_name = "facebook/nllb-200-distilled-600M"  # Default model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 512
        
        # Set HuggingFace cache directory if provided
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            logger.info(f"Using custom cache directory: {cache_dir}")
        
        # Initialize translation cache
        self.cache = TranslationCache(cache_dir)

        # Check if model is already cached
        self._check_and_load_model()

    def _check_and_load_model(self):
        """Check if model is cached and load it"""
        try:
            # Disable the download progress bar
            tf_utils.logging.set_verbosity_error()
            
            # Get and validate cache directory
            cache_dir = os.getenv('TRANSFORMERS_CACHE')
            if not cache_dir:
                logger.error("TRANSFORMERS_CACHE environment variable not set")
                raise ValueError("TRANSFORMERS_CACHE environment variable not set")

            # Construct model directory path
            model_dir = os.path.join(cache_dir, "models--facebook--nllb-200-distilled-600M", "snapshots", "main")
            logger.info(f"Looking for model files in: {model_dir}")

            # Check required files
            required_files = [
                "config.json",
                "pytorch_model.bin",
                "tokenizer.json",
                "tokenizer_config.json"
            ]
            
            missing_files = []
            for file in required_files:
                file_path = os.path.join(model_dir, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
                else:
                    logger.info(f"Found {file}")

            if missing_files:
                logger.error(f"Missing required model files: {', '.join(missing_files)}")
                raise FileNotFoundError(f"Missing required model files: {', '.join(missing_files)}")

            # Force offline mode
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'

            # Load model and tokenizer in offline mode
            logger.info("Loading model and tokenizer from cache...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=True,
                use_fast=True
            )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_dir,
                local_files_only=True
            ).to(self.device)
            
            logger.info("Translation model loaded successfully from cache")
            
        except Exception as e:
            logger.error(f"Failed to load translation model: {str(e)}")
            if isinstance(e, FileNotFoundError):
                logger.error("Please ensure all model files are in the correct location")
            raise

    @property
    def is_model_cached(self) -> bool:
        """Check if the model is already cached locally"""
        try:
            return self.tokenizer.from_pretrained(
                self.model_name,
                local_files_only=True,
                return_attention_mask=False
            ) is not None
        except:
            return False

    def get_model_cache_info(self) -> Dict:
        """Get information about the model cache"""
        cache_dir = os.getenv('TRANSFORMERS_CACHE', os.path.expanduser('~/.cache/huggingface'))
        cache_path = Path(cache_dir) / self.model_name.replace('/', '--')
        
        return {
            'is_cached': self.is_model_cached,
            'cache_directory': str(cache_dir),
            'model_path': str(cache_path),
            'device': self.device,
            'model_name': self.model_name
        }

    def validate_language_pair(self, source_lang: str, target_lang: str) -> bool:
        """Validate that the language pair is supported"""
        if source_lang not in self.SUPPORTED_LANGUAGES or target_lang not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language pair. Only English ↔ Indonesian translation is supported. "
                f"Got: {source_lang} → {target_lang}"
            )
        if source_lang == target_lang:
            raise ValueError("Source and target languages must be different")
        return True

    @lru_cache(maxsize=1000)
    def detect_language(self, text: str) -> str:
        """
        Detect if the text is English or Indonesian

        Args:
            text: Text to analyze

        Returns:
            Language code (eng_Latn or ind_Latn)
        """
        try:
            lang_code = langdetect.detect(text)
            
            if lang_code == "en":
                return "eng_Latn"
            elif lang_code == "id":
                return "ind_Latn"
            else:
                logger.warning(f"Detected unsupported language {lang_code}, defaulting to English")
                return "eng_Latn"
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            return "eng_Latn"

    def translate_text(self, text: str, source_language: Optional[str] = None,
                      target_language: str = "ind_Latn", preserve_layout: bool = True) -> str:
        """
        Translate text between English and Indonesian

        Args:
            text: Text to translate
            source_language: Source language code (eng_Latn or ind_Latn) or None for auto-detection
            target_language: Target language code (eng_Latn or ind_Latn)
            preserve_layout: Whether to preserve text layout (newlines, spaces)

        Returns:
            Translated text
        """
        if not text.strip():
            return text

        try:
            # Detect source language if not provided
            if not source_language:
                source_language = self.detect_language(text)
                logger.info(f"Auto-detected source language: {source_language}")

            # Validate language pair
            self.validate_language_pair(source_language, target_language)

            # Check cache first
            cached_translation = self.cache.get(text, source_language, target_language)
            if cached_translation:
                logger.debug("Using cached translation")
                return cached_translation

            if preserve_layout:
                # Split text into segments while preserving layout
                segments = self._split_preserve_layout(text)
                translated_segments = []

                for segment, layout_info in segments:
                    if not segment.strip():
                        translated_segments.append((segment, layout_info))
                        continue

                    # Tokenize the input
                    inputs = self.tokenizer(segment, return_tensors="pt", padding=True,
                                          truncation=True, max_length=self.max_length).to(self.device)

                    # Set the language tokens
                    self.tokenizer.src_lang = source_language
                    # Get the token ID for the target language
                    try:
                        forced_bos_token = target_language
                        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(forced_bos_token)
                        logger.info(f"Using target language token: {forced_bos_token} (ID: {forced_bos_token_id})")
                        
                        if forced_bos_token_id == self.tokenizer.unk_token_id:
                            logger.warning(f"Target language token {target_language} was converted to unknown token")
                            # Try with __${target_language}__ format
                            forced_bos_token = f"__{target_language}__"
                            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(forced_bos_token)
                            logger.info(f"Retrying with token: {forced_bos_token} (ID: {forced_bos_token_id})")
                    except Exception as e:
                        logger.error(f"Error setting target language token: {str(e)}")
                        raise

                    # Generate translation
                    translated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=self.max_length,
                        num_beams=5,
                        length_penalty=1.0
                    )

                    # Decode the output
                    translated_segment = self.tokenizer.batch_decode(
                        translated_tokens, skip_special_tokens=True)[0]
                    translated_segments.append((translated_segment, layout_info))

                # Reconstruct the text with original layout
                translated_text = self._reconstruct_layout(translated_segments)
            else:
                # Translate as a single block
                inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                      truncation=True, max_length=self.max_length).to(self.device)
                
                self.tokenizer.src_lang = source_language
                # Get the token ID for the target language
                try:
                    forced_bos_token = target_language
                    forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(forced_bos_token)
                    logger.info(f"Using target language token: {forced_bos_token} (ID: {forced_bos_token_id})")
                    
                    if forced_bos_token_id == self.tokenizer.unk_token_id:
                        logger.warning(f"Target language token {target_language} was converted to unknown token")
                        # Try with __${target_language}__ format
                        forced_bos_token = f"__{target_language}__"
                        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(forced_bos_token)
                        logger.info(f"Retrying with token: {forced_bos_token} (ID: {forced_bos_token_id})")
                except Exception as e:
                    logger.error(f"Error setting target language token: {str(e)}")
                    raise
                
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=self.max_length,
                    num_beams=5,
                    length_penalty=1.0
                )
                
                translated_text = self.tokenizer.batch_decode(
                    translated_tokens, skip_special_tokens=True)[0]

            # Cache the translation
            self.cache.set(text, source_language, target_language, translated_text)
            return translated_text

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text  # Return original text on error

    def _split_preserve_layout(self, text: str) -> List[Tuple[str, Dict]]:
        """Split text into segments while preserving layout information"""
        segments = []
        lines = text.split('\n')
        
        for line in lines:
            # Detect indentation
            leading_spaces = len(line) - len(line.lstrip())
            trailing_spaces = len(line) - len(line.rstrip())
            
            # Store layout information
            layout_info = {
                'leading_spaces': leading_spaces,
                'trailing_spaces': trailing_spaces,
                'newline': True  # Indicates this segment ends with a newline
            }
            
            segments.append((line, layout_info))
            
        return segments

    def _reconstruct_layout(self, segments: List[Tuple[str, Dict]]) -> str:
        """Reconstruct text from segments with preserved layout"""
        result = []
        
        for text, layout in segments:
            # Apply leading spaces
            line = ' ' * layout['leading_spaces'] + text.strip() + ' ' * layout['trailing_spaces']
            result.append(line)
            
        return '\n'.join(result)

    def translate_document(self, document: Dict[int, Dict], source_language: Optional[str] = None,
                         target_language: str = "ind_Latn", preserve_layout: bool = True) -> Dict[int, Dict]:
        """
        Translate an entire document (output from PDFExtractor)

        Args:
            document: Document dictionary from PDFExtractor
            source_language: Source language or None for auto-detection
            target_language: Target language
            preserve_layout: Whether to preserve document layout

        Returns:
            Translated document with the same structure
        """
        translated_document = {}

        # If we need to auto-detect, use the first page text for detection
        if not source_language and document:
            first_page = next(iter(document.values()))
            if 'text' in first_page and first_page['text']:
                source_language = self.detect_language(first_page['text'])
                logger.info(f"Auto-detected document language: {source_language}")

        # Collect all text segments for batch translation
        text_segments = []
        segment_mapping = []  # To track where each segment belongs

        # Process each page
        for page_num, page_content in document.items():
            translated_page = page_content.copy()
            translated_document[page_num] = translated_page

            # Add main text to batch
            if 'text' in page_content and page_content['text'].strip():
                text_segments.append(page_content['text'])
                segment_mapping.append(('text', page_num))

            # Add paragraphs to batch
            if 'paragraphs' in page_content and page_content['paragraphs']:
                for i, paragraph in enumerate(page_content['paragraphs']):
                    if paragraph.strip():
                        text_segments.append(paragraph)
                        segment_mapping.append(('paragraph', page_num, i))

            # Add table text to batch
            if 'tables' in page_content and page_content['tables']:
                for i, table in enumerate(page_content['tables']):
                    if 'text' in table and table['text'].strip():
                        text_segments.append(table['text'])
                        segment_mapping.append(('table', page_num, i))

        # Check cache for each segment first
        final_translations = []
        uncached_indices = []
        uncached_segments = []

        for i, segment in enumerate(text_segments):
            cached_translation = self.cache.get(segment, source_language, target_language)
            if cached_translation:
                final_translations.append(cached_translation)
            else:
                uncached_indices.append(i)
                uncached_segments.append(segment)

        # Batch translate uncached segments
        if uncached_segments:
            try:
                # Process in smaller batches to avoid memory issues
                batch_size = 10
                for i in range(0, len(uncached_segments), batch_size):
                    batch = uncached_segments[i:i + batch_size]
                    
                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length
                    ).to(self.device)

                    # Set source language
                    self.tokenizer.src_lang = source_language

                    # Get target language token
                    forced_bos_token = target_language
                    forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(forced_bos_token)

                    # Generate translations
                    translated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=self.max_length,
                        num_beams=5,
                        length_penalty=1.0
                    )

                    # Decode translations
                    batch_translations = self.tokenizer.batch_decode(
                        translated_tokens,
                        skip_special_tokens=True
                    )

                    # Cache the translations
                    for text, translation in zip(batch, batch_translations):
                        self.cache.set(text, source_language, target_language, translation)

                    # Insert translations at correct positions
                    for j, translation in enumerate(batch_translations):
                        idx = uncached_indices[i + j]
                        while len(final_translations) <= idx:
                            final_translations.append(None)
                        final_translations[idx] = translation

            except Exception as e:
                logger.error(f"Batch translation error: {str(e)}")
                return document

        # Map translations back to document structure
        for (segment_type, page_num, *extra), translation in zip(segment_mapping, final_translations):
            if translation is None:  # Skip failed translations
                continue
                
            if segment_type == 'text':
                translated_document[page_num]['text'] = translation
            elif segment_type == 'paragraph':
                if 'paragraphs' not in translated_document[page_num]:
                    translated_document[page_num]['paragraphs'] = page_content['paragraphs'].copy()
                translated_document[page_num]['paragraphs'][extra[0]] = translation
            elif segment_type == 'table':
                if 'tables' not in translated_document[page_num]:
                    translated_document[page_num]['tables'] = [table.copy() for table in page_content['tables']]
                translated_document[page_num]['tables'][extra[0]]['text'] = translation

        return translated_document