import fitz  # PyMuPDF

def extract_text_from_pdf(file_input) -> str:
    """
    Extracts text directly from a PDF file using PyMuPDF.
    Accepts either a string file path or bytes (from an uploaded file).
    """
    text = ""
    doc = None
    try:
        # Load from bytes if it's an uploaded file
        if isinstance(file_input, bytes):
            doc = fitz.open(stream=file_input, filetype="pdf")
        # Load from path if it's a string
        else:
            doc = fitz.open(file_input)
            
        # Iterate over pages to extract text
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    finally:
        if doc is not None:
            doc.close()
            
    return text.strip()
