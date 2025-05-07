import re
import logging

logger = logging.getLogger(__name__)

def detect_document_language(text: str) -> str:
    """
    Detect if a document is in Indonesian or English.

    Args:
        text: Sample text from the document

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