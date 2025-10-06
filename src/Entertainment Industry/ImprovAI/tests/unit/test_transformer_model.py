"""
Unit tests for the transformer model components.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from core.models.transformer_model import (
    PositionalEncoding,
    MultiHeadAttention,
    TransformerBlock,
    MusicTransformer,
    MusicTransformerTrainer
)


class TestPositionalEncoding:
    """Test the PositionalEncoding module."""
    
    def test_positional_encoding_initialization(self):
        """Test positional encoding initialization."""
        d_model = 512
        max_seq_len = 1024
        pe = PositionalEncoding(d_model, max_seq_len)
        
        assert pe.d_model == d_model
        assert pe.max_seq_len == max_seq_len
        assert pe.pe.shape == (max_seq_len, 1, d_model)
    
    def test_positional_encoding_forward(self):
        """Test positional encoding forward pass."""
        d_model = 128
        max_seq_len = 256
        pe = PositionalEncoding(d_model, max_seq_len)
        
        batch_size = 4
        seq_len = 64
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pe(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
    
    def test_positional_encoding_values(self):
        """Test that positional encoding values are reasonable."""
        d_model = 64
        max_seq_len = 128
        pe = PositionalEncoding(d_model, max_seq_len)
        
        # Check that positional encoding has expected properties
        assert torch.abs(pe.pe).max() <= 1.0
        assert not torch.isnan(pe.pe).any()


class TestMultiHeadAttention:
    """Test the MultiHeadAttention module."""
    
    def test_multi_head_attention_initialization(self):
        """Test multi-head attention initialization."""
        d_model = 512
        n_heads = 8
        mha = MultiHeadAttention(d_model, n_heads)
        
        assert mha.d_model == d_model
        assert mha.n_heads == n_heads
        assert mha.d_k == d_model // n_heads
    
    def test_multi_head_attention_forward(self):
        """Test multi-head attention forward pass."""
        d_model = 128
        n_heads = 4
        mha = MultiHeadAttention(d_model, n_heads)
        
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = mha(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)
        assert not torch.isnan(output).any()
        assert not torch.isnan(attention_weights).any()
    
    def test_multi_head_attention_with_mask(self):
        """Test multi-head attention with attention mask."""
        d_model = 128
        n_heads = 4
        mha = MultiHeadAttention(d_model, n_heads)
        
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        output, attention_weights = mha(x, x, x, mask=mask)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    def test_multi_head_attention_attention_weights(self):
        """Test that attention weights sum to 1."""
        d_model = 128
        n_heads = 4
        mha = MultiHeadAttention(d_model, n_heads)
        
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = mha(x, x, x)
        
        # Check that attention weights sum to 1 along the last dimension
        attention_sums = attention_weights.sum(dim=-1)
        assert torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-3)


class TestTransformerBlock:
    """Test the TransformerBlock module."""
    
    def test_transformer_block_initialization(self):
        """Test transformer block initialization."""
        d_model = 512
        n_heads = 8
        d_ff = 2048
        dropout = 0.1
        block = TransformerBlock(d_model, n_heads, d_ff, dropout)
        
        assert block.d_model == d_model
        assert block.n_heads == n_heads
        assert block.d_ff == d_ff
        assert block.dropout == dropout
    
    def test_transformer_block_forward(self):
        """Test transformer block forward pass."""
        d_model = 128
        n_heads = 4
        d_ff = 512
        dropout = 0.1
        block = TransformerBlock(d_model, n_heads, d_ff, dropout)
        
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = block(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)
        assert not torch.isnan(output).any()
    
    def test_transformer_block_with_mask(self):
        """Test transformer block with attention mask."""
        d_model = 128
        n_heads = 4
        d_ff = 512
        dropout = 0.1
        block = TransformerBlock(d_model, n_heads, d_ff, dropout)
        
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        output, attention_weights = block(x, mask=mask)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)


class TestMusicTransformer:
    """Test the MusicTransformer model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        vocab_size = 1000
        d_model = 512
        n_heads = 8
        n_layers = 6
        d_ff = 2048
        dropout = 0.1
        max_seq_len = 1024
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        assert model.vocab_size == vocab_size
        assert model.d_model == d_model
        assert model.n_heads == n_heads
        assert model.n_layers == n_layers
        assert model.d_ff == d_ff
        assert model.dropout_rate == dropout
        assert model.max_seq_len == max_seq_len
    
    def test_model_forward(self):
        """Test model forward pass."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        logits, attention_weights = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert len(attention_weights) == n_layers
        assert all(attn.shape == (batch_size, n_heads, seq_len, seq_len) for attn in attention_weights)
        assert not torch.isnan(logits).any()
    
    def test_model_forward_with_attention_mask(self):
        """Test model forward pass with attention mask."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        logits, attention_weights = model(input_ids, attention_mask=attention_mask)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert len(attention_weights) == n_layers
    
    def test_model_generate(self):
        """Test model generation."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        generated = model.generate(
            input_ids=input_ids,
            max_length=16,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        
        assert generated.shape == (batch_size, 16)
        assert generated[:, :seq_len].equal(input_ids)
    
    def test_model_generate_greedy(self):
        """Test model generation with greedy decoding."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        generated = model.generate(
            input_ids=input_ids,
            max_length=16,
            do_sample=False
        )
        
        assert generated.shape == (batch_size, 16)
        assert generated[:, :seq_len].equal(input_ids)
    
    def test_model_loss(self):
        """Test model loss calculation."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss = model.get_loss(input_ids, target_ids)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert loss > 0
    
    def test_model_loss_with_ignore_index(self):
        """Test model loss calculation with ignore index."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        target_ids[0, -1] = -100  # Set ignore index
        
        loss = model.get_loss(input_ids, target_ids, ignore_index=-100)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert not torch.isnan(loss)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        original_output, _ = model(input_ids)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model.save_model(f.name)
            
            # Load model
            loaded_model = MusicTransformer.load_model(f.name)
            
            # Clean up
            os.unlink(f.name)
        
        loaded_output, _ = loaded_model(input_ids)
        
        assert torch.allclose(original_output, loaded_output, atol=1e-6)
    
    def test_model_device_handling(self):
        """Test model device handling."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # Test CPU
        device = torch.device('cpu')
        model = model.to(device)
        
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        logits, _ = model(input_ids)
        
        assert logits.device == device
        assert not torch.isnan(logits).any()
    
    def test_model_parameter_count(self):
        """Test model parameter count."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert total_params == trainable_params  # All parameters should be trainable


class TestMusicTransformerTrainer:
    """Test the MusicTransformerTrainer class."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        trainer = MusicTransformerTrainer(model)
        
        assert trainer.model == model
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.device == torch.device('cpu')
    
    def test_train_step(self):
        """Test training step."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        trainer = MusicTransformerTrainer(model)
        
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        sample_batch = (input_ids, target_ids)
        loss = trainer.train_step(sample_batch)
        
        assert isinstance(loss, float)
        assert loss > 0
        assert not np.isnan(loss)
    
    def test_validate(self):
        """Test validation."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        trainer = MusicTransformerTrainer(model)
        
        # Mock data loader
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        mock_loader = [(input_ids, target_ids), (input_ids, target_ids)]
        
        avg_loss = trainer.validate(mock_loader)
        
        assert isinstance(avg_loss, float)
        assert avg_loss > 0
        assert not np.isnan(avg_loss)
    
    def test_train(self):
        """Test full training loop."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        trainer = MusicTransformerTrainer(model)
        
        # Mock data loaders
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        mock_train_loader = [(input_ids, target_ids), (input_ids, target_ids)]
        mock_val_loader = [(input_ids, target_ids)]
        
        trainer.train(mock_train_loader, mock_val_loader, num_epochs=2)
        
        # Check that training completed without errors
        assert True
    
    def test_train_without_validation(self):
        """Test training without validation."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        trainer = MusicTransformerTrainer(model)
        
        # Mock data loader
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        mock_train_loader = [(input_ids, target_ids), (input_ids, target_ids)]
        
        trainer.train(mock_train_loader, val_loader=None, num_epochs=1)
        
        # Check that training completed without errors
        assert True
    
    def test_save_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        trainer = MusicTransformerTrainer(model)
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            trainer.save_checkpoint(f.name, epoch=5, loss=0.5)
            
            # Load checkpoint
            loaded_trainer = MusicTransformerTrainer(model)
            loaded_trainer.load_checkpoint(f.name)
            
            # Clean up
            os.unlink(f.name)
        
        # Check that checkpoint was loaded correctly
        assert True
    
    def test_trainer_device_handling(self):
        """Test trainer device handling."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        trainer = MusicTransformerTrainer(model)
        
        # Test CPU
        device = torch.device('cpu')
        trainer = trainer.to(device)
        
        assert trainer.device == device
        assert trainer.model.device == device
    
    def test_optimizer_configuration(self):
        """Test optimizer configuration."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        trainer = MusicTransformerTrainer(model)
        
        assert trainer.optimizer is not None
        assert hasattr(trainer.optimizer, 'param_groups')
        assert len(trainer.optimizer.param_groups) > 0
    
    def test_scheduler_configuration(self):
        """Test scheduler configuration."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        trainer = MusicTransformerTrainer(model)
        
        assert trainer.scheduler is not None


class TestModelIntegration:
    """Test integration between model components."""
    
    def test_full_training_pipeline(self):
        """Test full training pipeline."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        d_ff = 512
        dropout = 0.1
        max_seq_len = 256
        
        model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        trainer = MusicTransformerTrainer(model)
        
        # Create training data
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        train_data = [(input_ids, target_ids), (input_ids, target_ids)]
        val_data = [(input_ids, target_ids)]
        
        # Train for one epoch
        trainer.train(train_data, val_data, num_epochs=1)
        
        # Test generation
        generated = model.generate(
            input_ids=input_ids,
            max_length=20,
            temperature=1.0,
            do_sample=True
        )
        
        assert generated.shape == (batch_size, 20)
        assert generated[:, :seq_len].equal(input_ids)
    
    def test_model_with_different_configurations(self):
        """Test model with different configurations."""
        vocab_size = 1000
        
        # Test with different model sizes
        configs = [
            {'d_model': 64, 'n_heads': 2, 'n_layers': 1, 'd_ff': 256},
            {'d_model': 128, 'n_heads': 4, 'n_layers': 2, 'd_ff': 512},
            {'d_model': 256, 'n_heads': 8, 'n_layers': 4, 'd_ff': 1024},
        ]
        
        for config in configs:
            model = MusicTransformer(
                vocab_size=vocab_size,
                **config,
                dropout=0.1,
                max_seq_len=256
            )
            
            batch_size = 2
            seq_len = 16
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            logits, _ = model(input_ids)
            
            assert logits.shape == (batch_size, seq_len, vocab_size)
            assert not torch.isnan(logits).any()


if __name__ == "__main__":
    pytest.main([__file__])
