import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import torch.optim as optim
import numpy as np
import math
import os
import time
import gc
from torch import Tensor
import matplotlib.pyplot as plt
from cleaning import preprocess_data, preprocess_for_recommendation, simple_tokenizer, create_dataloader

# Function for scaled dot-product attention
def scaled_dot_product(q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
    attention = nn.functional.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values

# Class for attention head
class AttentionHead(nn.Module):
    def __init__(self, dim_input: int, dim_q: int, dim_k: int):
        super().__init__()
        self.linear_query = nn.Linear(dim_input, dim_q)
        self.linear_key = nn.Linear(dim_input, dim_k)
        self.linear_value = nn.Linear(dim_input, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        return scaled_dot_product(
            self.linear_query(query),
            self.linear_key(key),
            self.linear_value(value),
            mask
        )

# Class for multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_input: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_input, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear_out = nn.Linear(num_heads * dim_k, dim_input)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        head_outputs = [head(query, key, value, mask) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        return self.linear_out(concatenated)

# Function for feed forward network
def feed_forward(input_dimension: int = 512, intermediate_dimension: int = 2048) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dimension, intermediate_dimension),
        nn.ReLU(),
        nn.Linear(intermediate_dimension, input_dimension),
    )

# Class for residual connections
class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout_rate: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.norm(x + self.dropout(self.sublayer(x, *args, **kwargs)))

# Function for positional encoding
def position_encoding(seq_len, d_model, device):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(seq_len)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return torch.tensor(pos_encoding, dtype=torch.float32).to(device)

# Class for Transformer encoder layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim: int = 512, heads_count: int = 6, feedforward_dim: int = 2048, dropout_rate: float = 0.1):
        super().__init__()
        query_dim = key_dim = model_dim // heads_count if model_dim % heads_count == 0 else model_dim // heads_count + 1
        self.multi_head_attention = Residual(
            MultiHeadAttention(heads_count, model_dim, query_dim, key_dim),
            dimension=model_dim,
            dropout_rate=dropout_rate,
        )
        self.feedforward_network = Residual(
            feed_forward(model_dim, feedforward_dim),
            dimension=model_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, src: Tensor, mask: Tensor = None) -> Tensor:
        """
        前向传播
        参数:
            src: 输入张量
            mask: 掩码张量 (可选)
        返回:
            src: 编码器层的输出
        """
        src = self.multi_head_attention(src, src, src, mask)
        return self.feedforward_network(src)

# Class for Transformer decoder layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim: int = 512, num_heads: int = 6, feedforward_dim: int = 2048, dropout_rate: float = 0.1):
        super().__init__()
        head_dim = max(model_dim // num_heads, 1)
        self.self_attention = Residual(
            MultiHeadAttention(num_heads, model_dim, head_dim, head_dim),
            dimension=model_dim,
            dropout_rate=dropout_rate,
        )
        self.cross_attention = Residual(
            MultiHeadAttention(num_heads, model_dim, head_dim, head_dim),
            dimension=model_dim,
            dropout_rate=dropout_rate,
        )
        self.feed_forward = Residual(
            feed_forward(model_dim, feedforward_dim),
            dimension=model_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, target: Tensor, memory: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        target = self.self_attention(target, target, target, tgt_mask)
        target = self.cross_attention(target, memory, memory, src_mask)
        return self.feed_forward(target)

# Class for Transformer model
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, model_dim, num_heads, num_layers, hidden_dim, max_seq_len, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, model_dim)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, model_dim)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(model_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(model_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(model_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_masks(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)
        future_mask = torch.tril(torch.ones((seq_length, seq_length), device=tgt.device)).bool()
        combined_tgt_mask = tgt_mask & future_mask.unsqueeze(0).unsqueeze(1)
        return src_mask, combined_tgt_mask

    def encode(self, src, src_mask):
        src_embedded = self.dropout(self.encoder_embedding(src) + position_encoding(src.size(1), self.encoder_embedding.embedding_dim, src.device))
        enc_output = src_embedded
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        return enc_output

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        tgt_embedded = self.dropout(self.decoder_embedding(tgt) + position_encoding(tgt.size(1), self.decoder_embedding.embedding_dim, tgt.device))
        dec_output = tgt_embedded
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        return dec_output

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.create_masks(src, tgt)
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)
        return output

def load_pretrained_weights(transformer, pretrained_state_dict):
    model_state_dict = transformer.state_dict()
    
    print("Keys in pretrained state dict:")
    for key in pretrained_state_dict.keys():
        print(key)
    
    print("\nKeys in current model state dict:")
    for key in model_state_dict.keys():
        print(key)
    
    for name, param in pretrained_state_dict.items():
        if name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                model_state_dict[name].copy_(param)
            else:
                print(f"Shape mismatch for layer {name}: {model_state_dict[name].shape} vs {param.shape}")
        else:
            print(f"Layer {name} not found in current model")
    
    transformer.load_state_dict(model_state_dict)
    print("Pretrained weights loaded (where matching)")

# Function to generate recommendations
def get_recommendations(user_input, vocab, model, label_encoder, max_seq_len, device, top_k=5):
    model.eval()
    with torch.no_grad():
        user_input_encoded = [vocab.get(word, 0) for movie in user_input for word in simple_tokenizer(movie)]
        user_input_padded = user_input_encoded + [0] * (max_seq_len - len(user_input_encoded))
        input_tensor = torch.tensor(user_input_padded, dtype=torch.long).unsqueeze(0).to(device)

        output = model(input_tensor, input_tensor)
        top_k_indices = torch.topk(output[0, -1, :], top_k).indices.tolist()

        # Ensure the indices are within the valid range
        valid_indices = np.clip(top_k_indices, 0, len(label_encoder.classes_) - 1)
    
    movie_names = label_encoder.inverse_transform(valid_indices)
    return movie_names