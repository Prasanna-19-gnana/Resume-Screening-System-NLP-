import fitz  # PyMuPDF
import docx
import re
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """ Extract text directly from bytes using PyMuPDF. Clean hidden spaces. """
    text = ""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text += page_text + "\n"
        doc.close()
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
    return clean_extracted_text(text)

def extract_text_from_docx(file_bytes: bytes) -> str:
    from io import BytesIO
    text = ""
    try:
        doc_io = BytesIO(file_bytes)
        doc = docx.Document(doc_io)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        logger.error(f"Error reading DOCX: {e}")
    return clean_extracted_text(text)

def clean_extracted_text(text: str) -> str:
    """ Remove unprintable chars and excessive newlines while preserving structure """
    if not text: return ""
    
    # Remove weird non-ascii headers/footers or unprintable bytes safely keeping tech tokens
    text = text.replace('\xa0', ' ')
    
    # Collapse 3+ newlines into 2 to preserve section boundaries
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()
