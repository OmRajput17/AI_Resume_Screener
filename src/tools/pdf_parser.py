import fitz, pdfplumber
from pathlib import Path

class PDFParser:
    def __init__(self):
        self.methods = {
            'pymupdf': self.parse_pdf_pymupdf,
            'pdfplumber': self.parse_pdf_pdfplumber,
            'pypdf2': self.parse_pdf_pypdf2,
        }

    def parse_pdf_pymupdf(self, file_path: str) -> str:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
        
    def parse_pdf_pdfplumber(self, file_path: str) -> str:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
        
    def parse_pdf_pypdf2(self, file_path: str) -> str:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    
    def parse_resume_pdf(self, file_path: str) -> str:
        path = str(Path(file_path).resolve())
        for name, fn in self.methods.items():
            try:
                text = fn(path)
                if len(text.strip()) > 100:
                    return text
            except Exception:
                continue
        raise ValueError(f'All PDF parsers failed for {path}')

