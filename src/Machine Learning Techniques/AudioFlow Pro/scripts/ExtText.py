"""
AudioFlow Pro - Text Extraction Module
Advanced PDF text extraction with intelligent formatting preservation
"""

import logging
from typing import Optional, Tuple
from io import BytesIO
import re
from pathlib import Path
from datetime import datetime

# PDF processing imports
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine
from pdfminer.pdfpage import PDFPage

logger = logging.getLogger(__name__)

class TextExtractor:
    """
    Advanced PDF text extraction with intelligent formatting preservation.
    
    Features:
    - Intelligent text extraction with layout preservation
    - Handling of complex PDF layouts
    - Text cleaning and normalization
    - Error handling for corrupted PDFs
    - Support for multiple PDF formats
    """
    
    def __init__(self):
        """Initialize the text extractor."""
        self.resource_manager = PDFResourceManager()
        self.laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5,
            detect_vertical=True
        )
        
        logger.info("TextExtractor initialized")
    
    def extract_text(self, uploaded_file) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract text from uploaded PDF file with advanced processing.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (extracted_text, error_message)
        """
        try:
            if not uploaded_file:
                return None, "No file uploaded"
            
            logger.info(f"Extracting text from file: {uploaded_file.name}")
            
            # Read file content
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            # Validate PDF content
            if not self._is_valid_pdf(file_content):
                return None, "Invalid or corrupted PDF file"
            
            # Extract text with advanced processing
            extracted_text = self._extract_text_advanced(file_content)
            
            if not extracted_text:
                return None, "No text content found in PDF"
            
            # Clean and normalize text
            cleaned_text = self._clean_text(extracted_text)
            
            if not cleaned_text:
                return None, "No readable text content found"
            
            logger.info(f"Successfully extracted {len(cleaned_text)} characters")
            return cleaned_text, None
            
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return None, f"Error extracting text: {str(e)}"
    
    def _is_valid_pdf(self, content: bytes) -> bool:
        """
        Validate PDF content.
        
        Args:
            content: PDF file content as bytes
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            # Check PDF header
            if not content.startswith(b'%PDF'):
                return False
            
            # Try to parse PDF structure
            from pdfminer.pdfparser import PDFParser
            from pdfminer.pdfdocument import PDFDocument
            
            parser = PDFParser(BytesIO(content))
            document = PDFDocument(parser)
            
            return True
        except Exception:
            return False
    
    def _extract_text_advanced(self, content: bytes) -> str:
        """
        Extract text with advanced processing and layout preservation.
        
        Args:
            content: PDF file content as bytes
            
        Returns:
            Extracted text string
        """
        try:
            # Use advanced extraction with layout analysis
            output_string = BytesIO()
            
            # Configure extraction parameters
            laparams = LAParams(
                line_margin=0.5,
                word_margin=0.1,
                char_margin=2.0,
                boxes_flow=0.5,
                detect_vertical=True,
                all_texts=True
            )
            
            # Extract text with layout preservation
            extract_text_to_fp(
                BytesIO(content),
                output_string,
                laparams=laparams,
                output_type='text'
            )
            
            return output_string.getvalue().decode('utf-8', errors='ignore')
            
        except Exception as e:
            logger.warning(f"Advanced extraction failed, trying basic method: {e}")
            return self._extract_text_basic(content)
    
    def _extract_text_basic(self, content: bytes) -> str:
        """
        Basic text extraction as fallback.
        
        Args:
            content: PDF file content as bytes
            
        Returns:
            Extracted text string
        """
        try:
            output_string = BytesIO()
            laparams = LAParams()
            
            extract_text_to_fp(
                BytesIO(content),
                output_string,
                laparams=laparams
            )
            
            return output_string.getvalue().decode('utf-8', errors='ignore')
            
        except Exception as e:
            logger.error(f"Basic extraction failed: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\']', '', text)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def extract_text_with_metadata(self, uploaded_file) -> dict:
        """
        Extract text with additional metadata.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Dictionary with text and metadata
        """
        try:
            text, error = self.extract_text(uploaded_file)
            
            if error:
                return {
                    "text": None,
                    "error": error,
                    "metadata": {}
                }
            
            # Calculate metadata
            metadata = {
                "character_count": len(text),
                "word_count": len(text.split()),
                "line_count": len(text.split('\n')),
                "file_name": uploaded_file.name,
                "file_size": uploaded_file.size,
                "extraction_method": "advanced"
            }
            
            return {
                "text": text,
                "error": None,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")
            return {
                "text": None,
                "error": str(e),
                "metadata": {}
            }
    
    def get_extraction_stats(self) -> dict:
        """
        Get extraction statistics.
        
        Returns:
            Dictionary with extraction statistics
        """
        return {
            "extractor_type": "Advanced PDF Text Extractor",
            "supported_formats": ["PDF"],
            "features": [
                "Layout preservation",
                "Text cleaning",
                "Error handling",
                "Metadata extraction"
            ],
            "timestamp": str(datetime.now())
        }

# Legacy function for backward compatibility
def extract_text(uploaded_file):
    """
    Legacy text extraction function for backward compatibility.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (extracted_text, error_message)
    """
    extractor = TextExtractor()
    return extractor.extract_text(uploaded_file)
