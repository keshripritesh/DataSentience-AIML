import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, max_len: int, use_pos_encoding: bool = True,
                 n_decoder_layers: int = 8, device: str = 'cpu'):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.device = device
        self.max_len = max_len

        # Embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_decoder_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize parameters
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embedding
        x = self.src_embedding(src) * math.sqrt(self.d_model)

        # Positional encoding
        if self.use_pos_encoding:
            x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embedding
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # Positional encoding
        if self.use_pos_encoding:
            x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)

        # Decoder layers
        for layer in self.decoder_layers:
            x = layer(x, tgt_mask)

        return x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encode
        memory = self.encode(src, src_mask)

        # Decode
        output = self.decode(tgt, memory, tgt_mask)

        # Project to vocabulary
        return self.output_projection(output)

    def generate(self, src: torch.Tensor, src_mask: torch.Tensor, max_len: int = 1000,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            # Encode source
            memory = self.encode(src, src_mask)

            # Initialize target with start token
            batch_size = src.size(0)
            tgt = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)

            for _ in range(max_len):
                # Forward pass
                output = self.decode(tgt, memory)
                logits = self.output_projection(output[:, -1, :]) / temperature

                # Apply top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
                    logits = logits_filtered

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                # Append to target
                tgt = torch.cat([tgt, next_token], dim=1)

                # Check for end token (assuming 0 is end token)
                if next_token.item() == 0:
                    break

            return tgt
