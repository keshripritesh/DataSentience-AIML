"""
Configuration management for ImprovAI.
Handles all project settings, model parameters, and environment variables.
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    
    # LSTM Configuration
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.2
    
    # Transformer Configuration
    transformer_d_model: int = 512
    transformer_nhead: int = 8
    transformer_num_layers: int = 6
    transformer_dropout: float = 0.1
    
    # Training Configuration
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_sequence_length: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    
    # Generation Configuration
    generation_length: int = 64
    min_notes: int = 4
    max_notes: int = 128


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    
    sample_rate: int = 44100
    hop_length: int = 512
    n_fft: int = 2048
    n_mels: int = 128
    
    # MIDI Configuration
    midi_resolution: int = 480
    velocity_range: tuple = (1, 127)
    note_range: tuple = (21, 108)  # A0 to C8
    
    # Audio Export
    export_format: str = "wav"
    export_quality: int = 320


@dataclass
class UIConfig:
    """Configuration for user interface."""
    
    # Streamlit Configuration
    page_title: str = "ImprovAI - Advanced AI Music Improviser"
    page_icon: str = "🎵"
    layout: str = "wide"
    
    # Piano Interface
    piano_octaves: int = 5
    piano_start_note: int = 48  # C3
    key_width: int = 40
    key_height: int = 120
    
    # Visualization
    plot_height: int = 400
    plot_width: int = 800
    color_scheme: str = "viridis"


@dataclass
class Config:
    """Main configuration class."""
    
    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "models")
    examples_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "examples")
    
    # Configuration objects
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Environment
    debug: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.examples_dir.mkdir(exist_ok=True)
    
    @classmethod
    def from_file(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            return cls(**config_data)
        else:
            return cls()
    
    def save(self, config_path: Optional[str] = None) -> None:
        """Save configuration to YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        
        config_path = Path(config_path)
        config_data = {
            'model': {
                'lstm_hidden_size': self.model.lstm_hidden_size,
                'lstm_num_layers': self.model.lstm_num_layers,
                'lstm_dropout': self.model.lstm_dropout,
                'transformer_d_model': self.model.transformer_d_model,
                'transformer_nhead': self.model.transformer_nhead,
                'transformer_num_layers': self.model.transformer_num_layers,
                'transformer_dropout': self.model.transformer_dropout,
                'learning_rate': self.model.learning_rate,
                'batch_size': self.model.batch_size,
                'max_sequence_length': self.model.max_sequence_length,
                'temperature': self.model.temperature,
                'top_k': self.model.top_k,
                'top_p': self.model.top_p,
                'generation_length': self.model.generation_length,
                'min_notes': self.model.min_notes,
                'max_notes': self.model.max_notes,
            },
            'audio': {
                'sample_rate': self.audio.sample_rate,
                'hop_length': self.audio.hop_length,
                'n_fft': self.audio.n_fft,
                'n_mels': self.audio.n_mels,
                'midi_resolution': self.audio.midi_resolution,
                'velocity_range': self.audio.velocity_range,
                'note_range': self.audio.note_range,
                'export_format': self.audio.export_format,
                'export_quality': self.audio.export_quality,
            },
            'ui': {
                'page_title': self.ui.page_title,
                'page_icon': self.ui.page_icon,
                'layout': self.ui.layout,
                'piano_octaves': self.ui.piano_octaves,
                'piano_start_note': self.ui.piano_start_note,
                'key_width': self.ui.key_width,
                'key_height': self.ui.key_height,
                'plot_height': self.ui.plot_height,
                'plot_width': self.ui.plot_width,
                'color_scheme': self.ui.color_scheme,
            },
            'debug': self.debug,
            'log_level': self.log_level,
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)


# Global configuration instance
config = Config.from_file()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> None:
    """Update configuration with new values."""
    global config
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.audio, key):
            setattr(config.audio, key, value)
        elif hasattr(config.ui, key):
            setattr(config.ui, key, value)
    
    # Save updated configuration
    config.save()
