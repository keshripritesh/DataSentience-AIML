"""
Performance tests for ImprovAI components.
"""

import pytest
import torch
import time
import numpy as np

from core.encoders.music_encoder import Note, MusicEncoder
from core.models.lstm_model import MusicLSTM
from core.models.transformer_model import MusicTransformer
from utils.config import get_config


class TestModelPerformance:
    """Test model performance characteristics."""
    
    def test_lstm_inference_speed(self):
        """Test LSTM model inference speed."""
        model = MusicLSTM(vocab_size=1000, embedding_dim=128, hidden_size=256)
        model.eval()
        
        batch_size = 4
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_ids)
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        assert avg_time < 1.0  # Should be fast
    
    def test_transformer_inference_speed(self):
        """Test Transformer model inference speed."""
        model = MusicTransformer(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512
        )
        model.eval()
        
        batch_size = 4
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_ids)
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        assert avg_time < 1.0  # Should be fast
    
    def test_encoder_performance(self):
        """Test music encoder performance."""
        config = get_config()
        encoder = MusicEncoder(config)
        
        # Create sample notes
        notes = [
            Note(pitch=60, velocity=80, start_time=0.0, end_time=0.5),
            Note(pitch=62, velocity=80, start_time=0.5, end_time=1.0),
            Note(pitch=64, velocity=80, start_time=1.0, end_time=1.5),
            Note(pitch=65, velocity=80, start_time=1.5, end_time=2.0),
        ]
        
        # Measure encoding time
        start_time = time.time()
        for _ in range(100):
            _ = encoder.encode_notes(notes)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.1  # Should be very fast


if __name__ == "__main__":
    pytest.main([__file__])
