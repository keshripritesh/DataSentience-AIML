"""
AudioFlow Pro - Pipeline Module
Advanced document processing pipeline with caching and error handling
"""

import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import hashlib
import json
import time
from datetime import datetime, timedelta
from io import BytesIO

# Import local modules
from ExtText import TextExtractor
from TTS import AudioGenerator
from config import AudioFlowConfig

logger = logging.getLogger(__name__)

class AudioFlowPipeline:
    """
    Advanced pipeline for document-to-speech conversion with caching and error handling.
    
    Features:
    - Intelligent text extraction with formatting preservation
    - Multi-language speech synthesis
    - Caching for improved performance
    - Comprehensive error handling and logging
    - Quality optimization and validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AudioFlow pipeline.
        
        Args:
            config: Configuration dictionary with language, speech_rate, and quality settings
        """
        self.config = config
        self.text_extractor = TextExtractor()
        self.audio_generator = AudioGenerator()
        self.cache_dir = AudioFlowConfig.setup_cache_directory()
        
        logger.info(f"AudioFlow pipeline initialized with config: {config}")
    
    def process_file(self, uploaded_file) -> Tuple[Optional[BytesIO], Optional[str]]:
        """
        Process uploaded file through the complete pipeline.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (audio_file, error_message)
        """
        try:
            # Validate file
            validation_result = self._validate_file(uploaded_file)
            if validation_result[1]:
                return None, validation_result[1]
            
            # Check cache
            cache_key = self._generate_cache_key(uploaded_file)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.info("Returning cached result")
                return cached_result, None
            
            # Extract text
            logger.info("Extracting text from PDF")
            extracted_text, error_message = self.text_extractor.extract_text(uploaded_file)
            if error_message:
                return None, error_message
            
            # Validate and clean text
            cleaned_text = self._clean_and_validate_text(extracted_text)
            if not cleaned_text:
                return None, AudioFlowConfig.get_error_message("text_too_short")
            
            # Generate audio
            logger.info("Generating audio from text")
            audio_file, error_message = self.audio_generator.generate_audio(
                cleaned_text, 
                self.config["language"],
                self.config["speech_rate"],
                self.config["quality"]
            )
            if error_message:
                return None, error_message
            
            # Cache the result
            self._save_to_cache(cache_key, audio_file)
            
            logger.info("Pipeline processing completed successfully")
            return audio_file, None
            
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            return None, f"Processing error: {str(e)}"
    
    def _validate_file(self, uploaded_file) -> Tuple[bool, Optional[str]]:
        """
        Validate uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not uploaded_file:
            return False, "No file uploaded"
        
        # Check file size
        if not AudioFlowConfig.validate_file_size(uploaded_file.size):
            return False, AudioFlowConfig.get_error_message("file_too_large")
        
        # Check file type
        if not uploaded_file.name.lower().endswith('.pdf'):
            return False, AudioFlowConfig.get_error_message("unsupported_format")
        
        return True, None
    
    def _clean_and_validate_text(self, text: str) -> Optional[str]:
        """
        Clean and validate extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text or None if invalid
        """
        if not text:
            return None
        
        # Remove extra whitespace
        cleaned_text = ' '.join(text.split())
        
        # Remove special characters that might cause TTS issues
        import re
        cleaned_text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', cleaned_text)
        
        # Validate length
        if not AudioFlowConfig.validate_text_length(len(cleaned_text)):
            return None
        
        return cleaned_text
    
    def _generate_cache_key(self, uploaded_file) -> str:
        """
        Generate cache key based on file content and configuration.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Cache key string
        """
        # Create hash from file content and config
        content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer
        
        config_hash = hashlib.md5(
            json.dumps(self.config, sort_keys=True).encode()
        ).hexdigest()
        
        content_hash = hashlib.md5(content).hexdigest()
        
        return f"{content_hash}_{config_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[BytesIO]:
        """
        Retrieve result from cache.
        
        Args:
            cache_key: Cache key string
            
        Returns:
            Cached audio file or None
        """
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        
        if cache_file.exists():
            # Check if cache is still valid
            if time.time() - cache_file.stat().st_mtime < AudioFlowConfig.CACHE_EXPIRY:
                try:
                    with open(cache_file, 'rb') as f:
                        audio_data = BytesIO(f.read())
                        logger.info("Retrieved result from cache")
                        return audio_data
                except Exception as e:
                    logger.warning(f"Failed to read cache file: {e}")
                    cache_file.unlink(missing_ok=True)
        
        return None
    
    def _save_to_cache(self, cache_key: str, audio_file: BytesIO):
        """
        Save result to cache.
        
        Args:
            cache_key: Cache key string
            audio_file: Audio file to cache
        """
        try:
            cache_file = self.cache_dir / f"{cache_key}.mp3"
            audio_file.seek(0)
            
            with open(cache_file, 'wb') as f:
                f.write(audio_file.read())
            
            logger.info("Saved result to cache")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        cache_files = list(self.cache_dir.glob("*.mp3"))
        
        return {
            "cache_size": len(cache_files),
            "cache_dir": str(self.cache_dir),
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_cache(self) -> bool:
        """
        Clear all cached results.
        
        Returns:
            True if cache cleared successfully
        """
        try:
            for cache_file in self.cache_dir.glob("*.mp3"):
                cache_file.unlink()
            logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

# Legacy function for backward compatibility
def pipeline(uploaded_file, lang='en'):
    """
    Legacy pipeline function for backward compatibility.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        lang: Language code
        
    Returns:
        Tuple of (audio_file, error_message)
    """
    config = {
        "language": lang,
        "speech_rate": 1.0,
        "quality": "Standard"
    }
    
    pipeline_instance = AudioFlowPipeline(config)
    return pipeline_instance.process_file(uploaded_file)
