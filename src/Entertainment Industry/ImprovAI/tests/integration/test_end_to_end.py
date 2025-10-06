"""
Integration tests for the full ImprovAI pipeline.
"""

import pytest
import torch
import numpy as np
import tempfile
import os

from core.encoders.music_encoder import Note, MusicalSequence, MusicEncoder
from core.models.lstm_model import MusicLSTM
from core.models.transformer_model import MusicTransformer
from core.generators.music_generator import MusicGenerator, GenerationConfig
# from io.midi_handler import MIDIHandler  # Commented out due to import conflict
from utils.config import get_config


class TestEndToEndPipeline:
    """Test the complete end-to-end pipeline."""
    
    def test_lstm_pipeline(self):
        """Test complete LSTM-based pipeline."""
        # Create sample notes
        notes = [
            Note(pitch=60, velocity=80, start_time=0.0, end_time=0.5),
            Note(pitch=62, velocity=80, start_time=0.5, end_time=1.0),
        ]
        
        # Initialize components
        config = get_config()
        encoder = MusicEncoder(config)
        model = MusicLSTM(vocab_size=100, embedding_dim=64, hidden_size=128)
        generator = MusicGenerator(config)
        
        # Encode notes
        encoded_sequence = encoder.encode_notes(notes)
        assert encoded_sequence.shape[0] > 0
        
        # Test generation
        generation_config = GenerationConfig(
            model_type="lstm",
            temperature=1.0,
            creativity=0.8,
            style="classical",
            tempo=120.0,
            key_signature="C"
        )
        
        continuation = generator.generate_continuation(notes, generation_config)
        assert isinstance(continuation, MusicalSequence)
    
    def test_transformer_pipeline(self):
        """Test complete Transformer-based pipeline."""
        # Create sample notes
        notes = [
            Note(pitch=60, velocity=80, start_time=0.0, end_time=0.5),
            Note(pitch=62, velocity=80, start_time=0.5, end_time=1.0),
        ]
        
        # Initialize components
        config = get_config()
        encoder = MusicEncoder(config)
        model = MusicTransformer(vocab_size=100, d_model=64, n_heads=2, n_layers=1)
        generator = MusicGenerator(config)
        
        # Encode notes
        encoded_sequence = encoder.encode_notes(notes)
        assert encoded_sequence.shape[0] > 0
        
        # Test generation
        generation_config = GenerationConfig(
            model_type="transformer",
            temperature=1.0,
            creativity=0.8,
            style="classical",
            tempo=120.0,
            key_signature="C"
        )
        
        continuation = generator.generate_continuation(notes, generation_config)
        assert isinstance(continuation, MusicalSequence)


if __name__ == "__main__":
    pytest.main([__file__])
