"""
Models package for ImprovAI.
"""

from .lstm_model import MusicLSTM, MusicLSTMTrainer, AttentionLayer
from .transformer_model import MusicTransformer, MusicTransformerTrainer

__all__ = [
    'MusicLSTM',
    'MusicLSTMTrainer', 
    'AttentionLayer',
    'MusicTransformer',
    'MusicTransformerTrainer'
]
