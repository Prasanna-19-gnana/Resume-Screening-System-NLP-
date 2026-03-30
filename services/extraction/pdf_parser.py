import pdfplumber
import io

def extract_text_from_pdf(file_input) -> str:
    """
    Extracts text directly from a PDF file using pdfplumber.
    Accepts either a string file path or bytes (from an uploaded file).
    """
    text = ""
    try:
        # Load from bytes if it's an uploaded file
        if isinstance(file_input, bytes):
            pdf = pdfplumber.open(io.BytesIO(file_input))
        # Load from path if it's a string
        else:
            pdf = pdfplumber.open(file_input)
            
        # Iterate over pages to extract text
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        pdf.close()
    except Exception as e:
        print(f"Error reading PDF: {e}")
            
    return text.strip()
