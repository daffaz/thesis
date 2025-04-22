# translation_service.py
from typing import Dict, List, Optional
import os
from pathlib import Path
import logging

# Import the NLLB model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import langdetect

logger = logging.getLogger(__name__)

class TranslationService:
    """
    Handles translation of text between languages using the NLLB model
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the translation service with an NLLB model

        Args:
            model_path: Path to a local NLLB model, or None to download from HF
        """
        self.model_name = model_path or "facebook/nllb-200-distilled-600M"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 512  # Adjust based on your model and memory constraints

        logger.info(f"Initializing translation model {self.model_name} on {self.device}")

        # Load model and tokenizer (might take some time)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

        # NLLB language code mapping (abbreviated)
        self.language_codes = {
            "english": "eng_Latn",
            "indonesian": "ind_Latn",
            # Add other languages as needed
        }

        logger.info("Translation model loaded successfully")

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text

        Args:
            text: Text to analyze

        Returns:
            Detected language code for NLLB
        """
        try:
            # Use langdetect for basic language detection
            lang_code = langdetect.detect(text)

            # Map to NLLB format (this is simplified - expand as needed)
            if lang_code == "en":
                return "eng_Latn"
            elif lang_code == "id":
                return "ind_Latn"
            else:
                logger.warning(f"Detected language {lang_code} not explicitly mapped, using English as fallback")
                return "eng_Latn"
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            return "eng_Latn"  # Default to English

    def translate_text(self, text: str, source_language: Optional[str] = None,
                       target_language: str = "ind_Latn") -> str:
        """
        Translate text from source language to target language

        Args:
            text: Text to translate
            source_language: Source language code (NLLB format) or None for auto-detection
            target_language: Target language code (NLLB format)

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

            # Convert friendly language names to NLLB codes if needed
            if source_language.lower() in self.language_codes:
                source_language = self.language_codes[source_language.lower()]

            if target_language.lower() in self.language_codes:
                target_language = self.language_codes[target_language.lower()]

            # Tokenize the input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                                   max_length=self.max_length).to(self.device)

            # Set the language tokens
            self.tokenizer.src_lang = source_language
            forced_bos_token_id = self.tokenizer.lang_code_to_id[target_language]

            # Generate translation
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=self.max_length,
                num_beams=5,
                length_penalty=1.0
            )

            # Decode the output
            translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

            return translated_text

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text  # Return original text on error

    def translate_document(self, document: Dict[int, Dict], source_language: Optional[str] = None,
                          target_language: str = "ind_Latn") -> Dict[int, Dict]:
        """
        Translate an entire document (output from PDFExtractor)

        Args:
            document: Document dictionary from PDFExtractor
            source_language: Source language or None for auto-detection
            target_language: Target language

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

        # Process each page
        for page_num, page_content in document.items():
            # Create a copy of the page content
            translated_page = page_content.copy()

            # Translate main text
            if 'text' in page_content and page_content['text']:
                translated_text = self.translate_text(
                    page_content['text'],
                    source_language,
                    target_language
                )
                translated_page['text'] = translated_text

            # Translate paragraphs if available
            if 'paragraphs' in page_content and page_content['paragraphs']:
                translated_paragraphs = []
                for paragraph in page_content['paragraphs']:
                    if paragraph.strip():
                        translated_para = self.translate_text(
                            paragraph,
                            source_language,
                            target_language
                        )
                        translated_paragraphs.append(translated_para)
                    else:
                        translated_paragraphs.append(paragraph)
                translated_page['paragraphs'] = translated_paragraphs

            # Translate tables if available
            if 'tables' in page_content and page_content['tables']:
                translated_tables = []
                for table in page_content['tables']:
                    translated_table = table.copy()
                    if 'text' in table and table['text'].strip():
                        translated_table_text = self.translate_text(
                            table['text'],
                            source_language,
                            target_language
                        )
                        translated_table['text'] = translated_table_text
                    translated_tables.append(translated_table)
                translated_page['tables'] = translated_tables

            # Add to the translated document
            translated_document[page_num] = translated_page

        return translated_document