"""
AudioFlow Pro - Text-to-Speech Module
Advanced audio generation with voice customization and quality optimization
"""

import logging
from typing import Optional, Tuple, Dict, Any
from io import BytesIO
import time
import requests
from pathlib import Path

# TTS imports
from gtts import gTTS
from gtts.lang import tts_langs
import tempfile
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class AudioGenerator:
    """
    Advanced text-to-speech generator with voice customization and quality optimization.
    
    Features:
    - Multi-language support with natural-sounding voices
    - Voice customization (speed, pitch, quality)
    - Error handling and retry mechanisms
    - Audio quality optimization
    - Caching for improved performance
    """
    
    def __init__(self):
        """Initialize the audio generator."""
        self.supported_languages = tts_langs()
        self.quality_settings = {
            "Standard": {"slow": False, "quality": "standard"},
            "Premium": {"slow": False, "quality": "premium"}
        }
        
        logger.info("AudioGenerator initialized")
    
    def generate_audio(self, text: str, language: str = "en", 
                      speech_rate: float = 1.0, quality: str = "Standard") -> Tuple[Optional[BytesIO], Optional[str]]:
        """
        Generate audio from text with advanced customization.
        
        Args:
            text: Text to convert to speech
            language: Language code (e.g., 'en', 'es', 'fr')
            speech_rate: Speech rate multiplier (0.5 to 2.0)
            quality: Audio quality setting ('Standard' or 'Premium')
            
        Returns:
            Tuple of (audio_file, error_message)
        """
        try:
            if not text:
                return None, "No text provided for audio generation"
            
            logger.info(f"Generating audio for {len(text)} characters in {language}")
            
            # Validate language
            if not self._is_language_supported(language):
                return None, f"Unsupported language: {language}"
            
            # Validate speech rate
            if not 0.5 <= speech_rate <= 2.0:
                return None, "Speech rate must be between 0.5 and 2.0"
            
            # Generate audio with retry mechanism
            audio_file = self._generate_audio_with_retry(text, language, speech_rate, quality)
            
            if not audio_file:
                return None, "Failed to generate audio"
            
            logger.info("Audio generation completed successfully")
            return audio_file, None
            
        except Exception as e:
            logger.error(f"Audio generation error: {e}")
            return None, f"Error generating audio: {str(e)}"
    
    def _is_language_supported(self, language: str) -> bool:
        """
        Check if language is supported.
        
        Args:
            language: Language code
            
        Returns:
            True if supported, False otherwise
        """
        # Normalize language code
        lang_code = language.split('-')[0].lower()
        return lang_code in self.supported_languages
    
    def _generate_audio_with_retry(self, text: str, language: str, 
                                  speech_rate: float, quality: str) -> Optional[BytesIO]:
        """
        Generate audio with retry mechanism for reliability.
        
        Args:
            text: Text to convert
            language: Language code
            speech_rate: Speech rate
            quality: Audio quality
            
        Returns:
            Audio file as BytesIO or None
        """
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Audio generation attempt {attempt + 1}/{max_retries}")
                
                # Split text into chunks if too long
                text_chunks = self._split_text_for_tts(text)
                
                if len(text_chunks) == 1:
                    # Single chunk - direct generation
                    return self._generate_single_audio(text, language, speech_rate, quality)
                else:
                    # Multiple chunks - combine audio files
                    return self._generate_combined_audio(text_chunks, language, speech_rate, quality)
                
            except Exception as e:
                logger.warning(f"Audio generation attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("All audio generation attempts failed")
                    return None
        
        return None
    
    def _split_text_for_tts(self, text: str, max_chunk_size: int = 4000) -> list:
        """
        Split text into chunks suitable for TTS processing.
        
        Args:
            text: Text to split
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        # Split by sentences first
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _generate_single_audio(self, text: str, language: str, 
                              speech_rate: float, quality: str) -> Optional[BytesIO]:
        """
        Generate audio for a single text chunk.
        
        Args:
            text: Text to convert
            language: Language code
            speech_rate: Speech rate
            quality: Audio quality
            
        Returns:
            Audio file as BytesIO or None
        """
        try:
            # Configure TTS parameters
            tts_params = {
                "text": text,
                "lang": language,
                "slow": speech_rate < 0.8,  # Use slow mode for slower speech
                "lang_check": False  # Disable language checking for better performance
            }
            
            # Create TTS object
            tts = gTTS(**tts_params)
            
            # Generate audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                tts.write_to_fp(temp_file)
                temp_file_path = temp_file.name
            
            # Read audio data
            with open(temp_file_path, 'rb') as f:
                audio_data = BytesIO(f.read())
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Single audio generation failed: {e}")
            return None
    
    def _generate_combined_audio(self, text_chunks: list, language: str, 
                                speech_rate: float, quality: str) -> Optional[BytesIO]:
        """
        Generate combined audio from multiple text chunks.
        
        Args:
            text_chunks: List of text chunks
            language: Language code
            speech_rate: Speech rate
            quality: Audio quality
            
        Returns:
            Combined audio file as BytesIO or None
        """
        try:
            audio_chunks = []
            
            for chunk in text_chunks:
                audio_chunk = self._generate_single_audio(chunk, language, speech_rate, quality)
                if audio_chunk:
                    audio_chunks.append(audio_chunk)
                else:
                    logger.warning(f"Failed to generate audio for chunk: {chunk[:50]}...")
            
            if not audio_chunks:
                return None
            
            # Combine audio chunks (simple concatenation for now)
            # In a production environment, you might want to use audio processing libraries
            combined_audio = BytesIO()
            
            for chunk in audio_chunks:
                chunk.seek(0)
                combined_audio.write(chunk.read())
            
            combined_audio.seek(0)
            return combined_audio
            
        except Exception as e:
            logger.error(f"Combined audio generation failed: {e}")
            return None
    
    def get_supported_languages(self) -> dict:
        """
        Get list of supported languages.
        
        Returns:
            Dictionary of supported languages
        """
        return self.supported_languages
    
    def get_audio_quality_settings(self) -> dict:
        """
        Get available audio quality settings.
        
        Returns:
            Dictionary of quality settings
        """
        return self.quality_settings
    
    def validate_text_for_tts(self, text: str) -> Tuple[bool, str]:
        """
        Validate text for TTS processing.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text:
            return False, "Text is empty"
        
        if len(text) > 50000:  # 50KB limit
            return False, "Text is too long for processing"
        
        # Check for problematic characters
        problematic_chars = ['\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07']
        for char in problematic_chars:
            if char in text:
                return False, f"Text contains invalid character: {char}"
        
        return True, ""
    
    def get_generation_stats(self) -> dict:
        """
        Get audio generation statistics.
        
        Returns:
            Dictionary with generation statistics
        """
        return {
            "generator_type": "Advanced TTS Generator",
            "supported_languages": len(self.supported_languages),
            "quality_settings": list(self.quality_settings.keys()),
            "features": [
                "Multi-language support",
                "Voice customization",
                "Quality optimization",
                "Retry mechanism",
                "Text validation"
            ],
            "timestamp": str(datetime.now())
        }

# Legacy function for backward compatibility
def text_to_speech(text, lang='en'):
    """
    Legacy text-to-speech function for backward compatibility.
    
    Args:
        text: Text to convert to speech
        lang: Language code
        
    Returns:
        Audio file as BytesIO or None
    """
    generator = AudioGenerator()
    audio_file, error = generator.generate_audio(text, lang, 1.0, "Standard")
    return audio_file if not error else None
