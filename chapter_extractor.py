import fitz 

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def split_by_custom_tags(text):
    """Split text using custom tags like @@CHAPTER_START@@."""
    sections = text.split("@@CHAPTER_START@@")
    return [section.strip() for section in sections if section.strip()]

def extract_chapters(pdf_path):
    """Extract chapters from a tagged PDF."""
    pdf_text = extract_text_from_pdf(pdf_path)
    return split_by_custom_tags(pdf_text)
