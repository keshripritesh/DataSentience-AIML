"""
AudioFlow Pro - Configuration Module
Centralized configuration management for the AudioFlow Pro platform
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AudioFlowConfig:
    """Configuration class for AudioFlow Pro settings."""
    
    # Supported languages with their codes
    SUPPORTED_LANGUAGES = {
        "English (US)": "en-us",
        "English (UK)": "en-gb",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Russian": "ru",
        "Japanese": "ja",
        "Korean": "ko",
        "Chinese (Simplified)": "zh-cn",
        "Chinese (Traditional)": "zh-tw",
        "Arabic": "ar",
        "Hindi": "hi",
        "Dutch": "nl",
        "Swedish": "sv",
        "Norwegian": "no",
        "Danish": "da",
        "Finnish": "fi",
        "Polish": "pl",
        "Czech": "cs",
        "Hungarian": "hu",
        "Romanian": "ro",
        "Bulgarian": "bg",
        "Croatian": "hr",
        "Slovak": "sk",
        "Slovenian": "sl",
        "Estonian": "et",
        "Latvian": "lv",
        "Lithuanian": "lt",
        "Greek": "el",
        "Turkish": "tr",
        "Hebrew": "he",
        "Thai": "th",
        "Vietnamese": "vi",
        "Indonesian": "id",
        "Malay": "ms",
        "Filipino": "tl",
        "Urdu": "ur",
        "Bengali": "bn",
        "Tamil": "ta",
        "Telugu": "te",
        "Marathi": "mr",
        "Gujarati": "gu",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Punjabi": "pa",
        "Assamese": "as",
        "Odia": "or",
        "Kashmiri": "ks",
        "Sindhi": "sd",
        "Nepali": "ne",
        "Sinhala": "si",
        "Burmese": "my",
        "Khmer": "km",
        "Lao": "lo",
        "Mongolian": "mn",
        "Tibetan": "bo",
        "Uyghur": "ug",
        "Kazakh": "kk",
        "Kyrgyz": "ky",
        "Tajik": "tg",
        "Turkmen": "tk",
        "Uzbek": "uz",
        "Azerbaijani": "az",
        "Georgian": "ka",
        "Armenian": "hy",
        "Persian": "fa",
        "Kurdish": "ku",
        "Pashto": "ps",
        "Dari": "prs",
        "Balochi": "bal",
        "Sindhi": "sd",
        "Saraiki": "skr",
        "Hindko": "hno",
        "Brahui": "brh",
        "Shina": "scl",
        "Khowar": "khw",
        "Wakhi": "wbl",
        "Burushaski": "bsk",
        "Kalasha": "kls",
        "Dameli": "dml",
        "Gawar-Bati": "gwt",
        "Phalura": "phl",
        "Kashmiri": "ks",
        "Dogri": "doi",
        "Kangri": "xnr",
        "Mandeali": "mjl",
        "Kullu": "kfx",
        "Kinnauri": "kfk",
        "Spiti": "spt",
        "Lahauli": "lbf",
        "Pahari": "phr",
        "Garhwali": "gbm",
        "Kumaoni": "kfy",
        "Jaunsari": "jns",
        "Sirmauri": "srx",
        "Baghati": "bfz",
        "Mandeali": "mjl",
        "Kangri": "xnr",
        "Kullu": "kfx",
        "Kinnauri": "kfk",
        "Spiti": "spt",
        "Lahauli": "lbf",
        "Pahari": "phr",
        "Garhwali": "gbm",
        "Kumaoni": "kfy",
        "Jaunsari": "jns",
        "Sirmauri": "srx",
        "Baghati": "bfz"
    }
    
    # File size limits
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MIN_FILE_SIZE = 1024  # 1KB
    
    # Audio quality settings
    AUDIO_QUALITY_SETTINGS = {
        "Standard": {
            "bitrate": "128k",
            "sample_rate": 22050,
            "channels": 1
        },
        "Premium": {
            "bitrate": "192k",
            "sample_rate": 44100,
            "channels": 2
        }
    }
    
    # Processing settings
    MAX_TEXT_LENGTH = 50000  # characters
    MIN_TEXT_LENGTH = 10     # characters
    
    # Cache settings
    CACHE_DIR = Path("cache")
    CACHE_EXPIRY = 3600  # 1 hour
    
    # API settings
    API_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    
    # Error messages
    ERROR_MESSAGES = {
        "file_too_large": "File size exceeds 100MB limit",
        "file_too_small": "File size is too small",
        "unsupported_format": "Unsupported file format",
        "extraction_failed": "Failed to extract text from PDF",
        "text_too_long": "Text is too long for processing",
        "text_too_short": "Text is too short for processing",
        "audio_generation_failed": "Failed to generate audio",
        "network_error": "Network connection error",
        "api_error": "API service error",
        "unknown_error": "An unknown error occurred"
    }
    
    # Success messages
    SUCCESS_MESSAGES = {
        "conversion_successful": "Document converted successfully",
        "text_extracted": "Text extracted successfully",
        "audio_generated": "Audio generated successfully"
    }
    
    @classmethod
    def get_language_code(cls, language_name: str) -> str:
        """Get language code from language name."""
        return cls.SUPPORTED_LANGUAGES.get(language_name, "en-us")
    
    @classmethod
    def get_language_name(cls, language_code: str) -> str:
        """Get language name from language code."""
        for name, code in cls.SUPPORTED_LANGUAGES.items():
            if code == language_code:
                return name
        return "English (US)"
    
    @classmethod
    def get_audio_settings(cls, quality: str) -> Dict:
        """Get audio quality settings."""
        return cls.AUDIO_QUALITY_SETTINGS.get(quality, cls.AUDIO_QUALITY_SETTINGS["Standard"])
    
    @classmethod
    def validate_file_size(cls, file_size: int) -> bool:
        """Validate file size."""
        return cls.MIN_FILE_SIZE <= file_size <= cls.MAX_FILE_SIZE
    
    @classmethod
    def validate_text_length(cls, text_length: int) -> bool:
        """Validate text length."""
        return cls.MIN_TEXT_LENGTH <= text_length <= cls.MAX_TEXT_LENGTH
    
    @classmethod
    def get_error_message(cls, error_type: str) -> str:
        """Get error message by type."""
        return cls.ERROR_MESSAGES.get(error_type, cls.ERROR_MESSAGES["unknown_error"])
    
    @classmethod
    def get_success_message(cls, success_type: str) -> str:
        """Get success message by type."""
        return cls.SUCCESS_MESSAGES.get(success_type, "Operation completed successfully")
    
    @classmethod
    def setup_cache_directory(cls) -> Path:
        """Setup cache directory."""
        cache_dir = cls.CACHE_DIR
        cache_dir.mkdir(exist_ok=True)
        return cache_dir 