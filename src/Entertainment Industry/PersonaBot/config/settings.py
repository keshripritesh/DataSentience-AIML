"""
Configuration settings for PersonaBot
Advanced conversational AI with dynamic personality adaptation
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for the NLP model"""
    model_name: str = "gpt2"
    max_length: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True

@dataclass
class PersonalityConfig:
    """Configuration for personality traits"""
    traits: List[str] = None
    default_values: Dict[str, float] = None
    
    def __post_init__(self):
        if self.traits is None:
            self.traits = [
                "humor", "formality", "empathy", "sarcasm",
                "enthusiasm", "professionalism", "creativity", "assertiveness"
            ]
        
        if self.default_values is None:
            self.default_values = {
                "humor": 0.5,
                "formality": 0.5,
                "empathy": 0.5,
                "sarcasm": 0.3,
                "enthusiasm": 0.6,
                "professionalism": 0.7,
                "creativity": 0.5,
                "assertiveness": 0.4
            }

@dataclass
class RLConfig:
    """Configuration for Reinforcement Learning"""
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    memory_size: int = 10000
    batch_size: int = 32
    update_frequency: int = 100

@dataclass
class RewardConfig:
    """Configuration for reward functions"""
    engagement_weight: float = 0.4
    sentiment_weight: float = 0.3
    relevance_weight: float = 0.2
    coherence_weight: float = 0.1
    max_conversation_length: int = 50
    min_response_length: int = 5

@dataclass
class UIConfig:
    """Configuration for user interfaces"""
    cli_prompt: str = "🤖 PersonaBot > "
    web_title: str = "PersonaBot - Advanced Conversational AI"
    web_description: str = "Dynamic personality adaptation using RL"
    max_display_messages: int = 20

@dataclass
class DataConfig:
    """Configuration for data storage"""
    sessions_dir: str = "data/sessions"
    models_dir: str = "data/models"
    logs_dir: str = "data/logs"
    session_format: str = "json"
    auto_save: bool = True
    save_frequency: int = 10

class Settings:
    """Main settings class that combines all configurations"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.personality = PersonalityConfig()
        self.rl = RLConfig()
        self.reward = RewardConfig()
        self.ui = UIConfig()
        self.data = DataConfig()
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.data.sessions_dir,
            self.data.models_dir,
            self.data.logs_dir,
            "tests",
            "core",
            "ui",
            "utils"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "model": self.model.__dict__,
            "personality": self.personality.__dict__,
            "rl": self.rl.__dict__,
            "reward": self.reward.__dict__,
            "ui": self.ui.__dict__,
            "data": self.data.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Settings':
        """Create settings from dictionary"""
        settings = cls()
        
        if "model" in config_dict:
            for key, value in config_dict["model"].items():
                setattr(settings.model, key, value)
        
        if "personality" in config_dict:
            for key, value in config_dict["personality"].items():
                setattr(settings.personality, key, value)
        
        if "rl" in config_dict:
            for key, value in config_dict["rl"].items():
                setattr(settings.rl, key, value)
        
        if "reward" in config_dict:
            for key, value in config_dict["reward"].items():
                setattr(settings.reward, key, value)
        
        if "ui" in config_dict:
            for key, value in config_dict["ui"].items():
                setattr(settings.ui, key, value)
        
        if "data" in config_dict:
            for key, value in config_dict["data"].items():
                setattr(settings.data, key, value)
        
        return settings

# Global settings instance
settings = Settings()

# Environment variables override
def load_env_settings():
    """Load settings from environment variables"""
    if os.getenv("PERSONABOT_MODEL_NAME"):
        settings.model.model_name = os.getenv("PERSONABOT_MODEL_NAME")
    
    if os.getenv("PERSONABOT_LEARNING_RATE"):
        settings.rl.learning_rate = float(os.getenv("PERSONABOT_LEARNING_RATE"))
    
    if os.getenv("PERSONABOT_SESSIONS_DIR"):
        settings.data.sessions_dir = os.getenv("PERSONABOT_SESSIONS_DIR")

# Load environment settings
load_env_settings()
