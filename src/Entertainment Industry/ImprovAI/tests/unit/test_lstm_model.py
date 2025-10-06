"""
Unit tests for the LSTM model module.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from core.models.lstm_model import (
    MusicLSTM, MusicLSTMTrainer, AttentionLayer
)
from utils.config import get_config


class TestAttentionLayer:
    """Test the AttentionLayer class."""
    
    @pytest.fixture
    def attention_layer(self):
        """Create an AttentionLayer instance for testing."""
        return AttentionLayer(hidden_size=256, attention_size=128)
    
    @pytest.fixture
    def sample_hidden_states(self):
        """Create sample hidden states for testing."""
        return torch.randn(2, 10, 256)  # batch_size=2, seq_len=10, hidden_size=256
    
    def test_attention_layer_initialization(self, attention_layer):
        """Test attention layer initialization."""
        assert attention_layer.hidden_size == 256
        assert attention_layer.attention_size == 128
        assert hasattr(attention_layer, 'attention_weights')
        assert hasattr(attention_layer, 'attention_context')
    
    def test_attention_forward(self, attention_layer, sample_hidden_states):
        """Test attention layer forward pass."""
        context_vector, attention_weights = attention_layer(sample_hidden_states)
        
        assert isinstance(context_vector, torch.Tensor)
        assert isinstance(attention_weights, torch.Tensor)
        assert context_vector.shape == (2, 256)  # batch_size, hidden_size
        assert attention_weights.shape == (2, 10)  # batch_size, seq_len
        assert torch.allclose(attention_weights.sum(dim=1), torch.ones(2))  # Sum to 1
    
    def test_attention_weights_sum_to_one(self, attention_layer, sample_hidden_states):
        """Test that attention weights sum to one for each sequence."""
        _, attention_weights = attention_layer(sample_hidden_states)
        
        # Check that weights sum to 1 for each sequence
        sums = attention_weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    
    def test_attention_with_different_batch_sizes(self, attention_layer):
        """Test attention layer with different batch sizes."""
        batch_sizes = [1, 4, 8]
        
        for batch_size in batch_sizes:
            hidden_states = torch.randn(batch_size, 10, 256)
            context_vector, attention_weights = attention_layer(hidden_states)
            
            assert context_vector.shape == (batch_size, 256)
            assert attention_weights.shape == (batch_size, 10)


class TestMusicLSTM:
    """Test the MusicLSTM class."""
    
    @pytest.fixture
    def model(self):
        """Create a MusicLSTM instance for testing."""
        return MusicLSTM(
            vocab_size=1000,
            embedding_dim=128,
            hidden_size=256,
            num_layers=2,
            dropout=0.1,
            use_attention=True,
            use_residual=True
        )
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input for testing."""
        return torch.randint(0, 1000, (4, 16))  # batch_size=4, seq_len=16
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.vocab_size == 1000
        assert model.embedding_dim == 128
        assert model.hidden_size == 256
        assert model.num_layers == 2
        assert model.dropout == 0.1
        assert model.use_attention == True
        assert model.use_residual == True
        
        # Check that components exist
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'lstm_layers')
        assert hasattr(model, 'attention')
        assert hasattr(model, 'output_projection')
        assert hasattr(model, 'batch_norm')
        assert hasattr(model, 'layer_norms')
    
    def test_model_forward(self, model, sample_input):
        """Test model forward pass."""
        logits, (final_hidden, attention_weights) = model(sample_input)
        
        assert isinstance(logits, torch.Tensor)
        assert isinstance(final_hidden, torch.Tensor)
        assert isinstance(attention_weights, torch.Tensor)
        
        # Check shapes
        assert logits.shape == (4, 16, 1000)  # batch_size, seq_len, vocab_size
        assert final_hidden.shape == (4, 256)  # batch_size, hidden_size
        assert attention_weights.shape == (4, 16)  # batch_size, seq_len
    
    def test_model_forward_without_attention(self):
        """Test model forward pass without attention."""
        model = MusicLSTM(
            vocab_size=1000,
            embedding_dim=128,
            hidden_size=256,
            num_layers=2,
            use_attention=False
        )
        
        sample_input = torch.randint(0, 1000, (4, 16))
        logits, (final_hidden, attention_weights) = model(sample_input)
        
        assert isinstance(logits, torch.Tensor)
        assert isinstance(final_hidden, torch.Tensor)
        assert attention_weights is None
        
        assert logits.shape == (4, 16, 1000)
        assert final_hidden.shape == (4, 256)
    
    def test_model_forward_without_residual(self):
        """Test model forward pass without residual connections."""
        model = MusicLSTM(
            vocab_size=1000,
            embedding_dim=128,
            hidden_size=256,
            num_layers=2,
            use_residual=False
        )
        
        sample_input = torch.randint(0, 1000, (4, 16))
        logits, _ = model(sample_input)
        
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (4, 16, 1000)
    
    def test_model_generate(self, model, sample_input):
        """Test model generation."""
        generated = model.generate(
            sample_input,
            max_length=8,
            temperature=1.0,
            do_sample=True
        )
        
        assert isinstance(generated, torch.Tensor)
        assert generated.shape == (4, 24)  # batch_size, seq_len + max_length
        assert torch.all(generated >= 0)
        assert torch.all(generated < 1000)
    
    def test_model_generate_greedy(self, model, sample_input):
        """Test model generation with greedy decoding."""
        generated = model.generate(
            sample_input,
            max_length=8,
            do_sample=False
        )
        
        assert isinstance(generated, torch.Tensor)
        assert generated.shape == (4, 24)
    
    def test_model_generate_with_top_k(self, model, sample_input):
        """Test model generation with top-k sampling."""
        generated = model.generate(
            sample_input,
            max_length=8,
            top_k=10,
            do_sample=True
        )
        
        assert isinstance(generated, torch.Tensor)
        assert generated.shape == (4, 24)
    
    def test_model_generate_with_top_p(self, model, sample_input):
        """Test model generation with nucleus sampling."""
        generated = model.generate(
            sample_input,
            max_length=8,
            top_p=0.9,
            do_sample=True
        )
        
        assert isinstance(generated, torch.Tensor)
        assert generated.shape == (4, 24)
    
    def test_model_loss(self, model, sample_input):
        """Test model loss calculation."""
        target_ids = torch.randint(0, 1000, (4, 16))
        
        loss = model.get_loss(sample_input, target_ids)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.item() > 0
    
    def test_model_loss_with_ignore_index(self, model, sample_input):
        """Test model loss calculation with ignore index."""
        target_ids = torch.randint(0, 1000, (4, 16))
        target_ids[0, 0] = -100  # Ignore this token
        
        loss = model.get_loss(sample_input, target_ids, ignore_index=-100)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.item() > 0
    
    def test_model_save_load(self, model, tmp_path):
        """Test model saving and loading."""
        # Save model
        save_path = tmp_path / "test_model.pth"
        model.save_model(str(save_path))
        
        assert save_path.exists()
        
        # Load model
        loaded_model = MusicLSTM.load_model(str(save_path))
        
        assert loaded_model.vocab_size == model.vocab_size
        assert loaded_model.embedding_dim == model.embedding_dim
        assert loaded_model.hidden_size == model.hidden_size
        assert loaded_model.num_layers == model.num_layers
        assert loaded_model.dropout == model.dropout
        assert loaded_model.use_attention == model.use_attention
        assert loaded_model.use_residual == model.use_residual
        
        # Test that loaded model produces same output
        sample_input = torch.randint(0, 1000, (2, 8))
        
        model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            original_output, _ = model(sample_input)
            loaded_output, _ = loaded_model(sample_input)
            
            assert torch.allclose(original_output, loaded_output, atol=1e-6)
    
    def test_model_device_handling(self, model):
        """Test model device handling."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model.to(device)
            
            sample_input = torch.randint(0, 1000, (2, 8)).to(device)
            logits, _ = model(sample_input)
            
            assert logits.device == device
        else:
            # Test CPU
            device = torch.device('cpu')
            model.to(device)
            
            sample_input = torch.randint(0, 1000, (2, 8))
            logits, _ = model(sample_input)
            
            assert logits.device == device
    
    def test_model_parameter_count(self, model):
        """Test that model has reasonable number of parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should have at least some parameters
        assert total_params > 0
        
        # Should not have too many parameters for a test model
        assert total_params < 10_000_000  # 10M parameters max for test
    
    def test_model_gradient_flow(self, model, sample_input):
        """Test that gradients flow through the model."""
        target_ids = torch.randint(0, 1000, (4, 16))
        
        loss = model.get_loss(sample_input, target_ids)
        loss.backward()
        
        # Check that gradients exist for some parameters
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        assert has_gradients


class TestMusicLSTMTrainer:
    """Test the MusicLSTMTrainer class."""
    
    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        return MusicLSTM(
            vocab_size=1000,
            embedding_dim=128,
            hidden_size=256,
            num_layers=2
        )
    
    @pytest.fixture
    def trainer(self, model):
        """Create a trainer instance for testing."""
        return MusicLSTMTrainer(model)
    
    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        input_ids = torch.randint(0, 1000, (4, 16))
        target_ids = torch.randint(0, 1000, (4, 16))
        return (input_ids, target_ids)
    
    def test_trainer_initialization(self, trainer, model):
        """Test trainer initialization."""
        assert trainer.model == model
        assert hasattr(trainer, 'optimizer')
        assert hasattr(trainer, 'scheduler')
        assert trainer.device in [torch.device('cpu'), torch.device('cuda')]
    
    def test_train_step(self, trainer, sample_batch):
        """Test training step."""
        loss = trainer.train_step(sample_batch)
        
        assert isinstance(loss, float)
        assert loss > 0
    
    def test_validate(self, trainer):
        """Test validation."""
        # Create mock data loader
        mock_loader = [
            (torch.randint(0, 1000, (2, 8)), torch.randint(0, 1000, (2, 8))),
            (torch.randint(0, 1000, (2, 8)), torch.randint(0, 1000, (2, 8)))
        ]
        
        avg_loss = trainer.validate(mock_loader)
        
        assert isinstance(avg_loss, float)
        assert avg_loss > 0
    
    def test_train(self, trainer):
        """Test training loop."""
        # Create mock data loaders
        mock_train_loader = [
            (torch.randint(0, 1000, (2, 8)), torch.randint(0, 1000, (2, 8))),
            (torch.randint(0, 1000, (2, 8)), torch.randint(0, 1000, (2, 8)))
        ]
        
        mock_val_loader = [
            (torch.randint(0, 1000, (2, 8)), torch.randint(0, 1000, (2, 8)))
        ]
        
        # Train for 2 epochs
        trainer.train(mock_train_loader, mock_val_loader, num_epochs=2)
        
        # Check that model parameters have been updated
        has_gradients = False
        for param in trainer.model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        assert has_gradients
    
    def test_train_without_validation(self, trainer):
        """Test training without validation."""
        mock_train_loader = [
            (torch.randint(0, 1000, (2, 8)), torch.randint(0, 1000, (2, 8))),
            (torch.randint(0, 1000, (2, 8)), torch.randint(0, 1000, (2, 8)))
        ]
        
        trainer.train(mock_train_loader, val_loader=None, num_epochs=1)
        
        # Should complete without errors
        assert True
    
    def test_save_load_checkpoint(self, trainer, tmp_path):
        """Test saving and loading checkpoints."""
        checkpoint_path = tmp_path / "test_checkpoint.pth"
        
        # Save checkpoint
        trainer.save_checkpoint(str(checkpoint_path))
        
        assert checkpoint_path.exists()
        
        # Load checkpoint
        trainer.load_checkpoint(str(checkpoint_path))
        
        # Should complete without errors
        assert True
    
    def test_trainer_device_handling(self, model):
        """Test trainer device handling."""
        trainer = MusicLSTMTrainer(model)
        
        # Model should be on the same device as trainer
        assert trainer.model.device == trainer.device
        
        # Test with sample batch
        sample_batch = (
            torch.randint(0, 1000, (2, 8)),
            torch.randint(0, 1000, (2, 8))
        )
        
        loss = trainer.train_step(sample_batch)
        assert isinstance(loss, float)
    
    def test_optimizer_configuration(self, trainer):
        """Test optimizer configuration."""
        optimizer = trainer.optimizer
        
        # Check that optimizer has parameters
        assert len(optimizer.param_groups) > 0
        
        # Check learning rate
        for group in optimizer.param_groups:
            assert 'lr' in group
            assert group['lr'] > 0
    
    def test_scheduler_configuration(self, trainer):
        """Test scheduler configuration."""
        scheduler = trainer.scheduler
        
        # Check that scheduler exists
        assert scheduler is not None
        
        # Check that it's a learning rate scheduler
        assert hasattr(scheduler, 'step')


class TestModelIntegration:
    """Integration tests for the LSTM model."""
    
    def test_full_training_pipeline(self):
        """Test a complete training pipeline."""
        # Create model
        model = MusicLSTM(
            vocab_size=1000,
            embedding_dim=64,
            hidden_size=128,
            num_layers=2
        )
        
        # Create trainer
        trainer = MusicLSTMTrainer(model)
        
        # Create training data
        train_data = [
            (torch.randint(0, 1000, (2, 8)), torch.randint(0, 1000, (2, 8))),
            (torch.randint(0, 1000, (2, 8)), torch.randint(0, 1000, (2, 8))),
            (torch.randint(0, 1000, (2, 8)), torch.randint(0, 1000, (2, 8)))
        ]
        
        val_data = [
            (torch.randint(0, 1000, (2, 8)), torch.randint(0, 1000, (2, 8)))
        ]
        
        # Train for one epoch
        trainer.train(train_data, val_data, num_epochs=1)
        
        # Test generation
        sample_input = torch.randint(0, 1000, (1, 4))
        generated = model.generate(sample_input, max_length=4)
        
        assert generated.shape == (1, 8)  # batch_size, seq_len + max_length
        assert torch.all(generated >= 0)
        assert torch.all(generated < 1000)
    
    def test_model_with_different_configurations(self):
        """Test model with different configurations."""
        configs = [
            {'use_attention': True, 'use_residual': True},
            {'use_attention': True, 'use_residual': False},
            {'use_attention': False, 'use_residual': True},
            {'use_attention': False, 'use_residual': False},
        ]
        
        for config in configs:
            model = MusicLSTM(
                vocab_size=1000,
                embedding_dim=64,
                hidden_size=128,
                num_layers=2,
                **config
            )
            
            sample_input = torch.randint(0, 1000, (2, 8))
            logits, _ = model(sample_input)
            
            assert logits.shape == (2, 8, 1000)
            assert torch.all(torch.isfinite(logits))


if __name__ == "__main__":
    pytest.main([__file__])
