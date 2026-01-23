"""
PDF Parser Module
Extracts text content from PDF files
"""

import pdfplumber
from PyPDF2 import PdfReader


def parse_pdf(filepath):
    """
    Parse PDF file and extract text content
    
    Args:
        filepath (str): Path to the PDF file
        
    Returns:
        str: Extracted text from PDF
    """
    text = ""
    
    try:
        # Method 1: Try pdfplumber first (better for complex PDFs)
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        
        # If pdfplumber fails or returns empty, try PyPDF2
        if not text.strip():
            with open(filepath, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        
        return text.strip()
    
    except Exception as e:
        raise Exception(f"Error parsing PDF: {str(e)}")


def parse_pdf_with_layout(filepath):
    """
    Parse PDF preserving layout information
    Useful for structured CVs
    
    Args:
        filepath (str): Path to the PDF file
        
    Returns:
        dict: Extracted text with layout information
    """
    try:
        result = {
            'text': '',
            'pages': [],
            'tables': []
        }
        
        with pdfplumber.open(filepath) as pdf:
            for i, page in enumerate(pdf.pages):
                page_data = {
                    'page_number': i + 1,
                    'text': page.extract_text() or '',
                    'tables': page.extract_tables() or []
                }
                
                result['pages'].append(page_data)
                result['text'] += page_data['text'] + "\n\n"
                
                if page_data['tables']:
                    result['tables'].extend(page_data['tables'])
        
        return result
    
    except Exception as e:
        raise Exception(f"Error parsing PDF with layout: {str(e)}")