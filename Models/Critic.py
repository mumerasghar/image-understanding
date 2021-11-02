import torch
import torch.nn as nn
import torch.nn.functional as F
from .SubLayers import MultiHeadAttention


class CriticFeedForward(nn.Module):
    def __init__(self, _input, _hidden, _output, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(_input, _hidden)
        self.w_2 = nn.Linear(_hidden, _output)
        self.norm = nn.LayerNorm(_input, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.sigmoid(self.w_2(self.dropout(self.norm(F.relu(self.w_1(x))))))
        return x


class Critic(nn.Module):

    def __init__(self, d_model, num_heads, d_k, d_v, dropout=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(num_heads, d_model, d_k, d_v, dropout=dropout)
        self.mha2 = MultiHeadAttention(num_heads, d_model, d_k, d_v, dropout=dropout)

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.fnn = CriticFeedForward(d_model, d_model, 1, dropout)

    def forward(self, x):
        x = x.to(torch.float32)
        att1, _ = self.mha1(x, x, x)
        out1 = self.dropout1(att1)

        att2, _ = self.mha2(x, out1, out1)
        out2 = self.dropout2(att2)

        ffn_output = self.fnn(out2)
        ffn_output = self.dropout3(ffn_output)

        return ffn_output
