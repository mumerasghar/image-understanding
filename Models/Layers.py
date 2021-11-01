import torch.nn as nn
import torch.nn.functional as F
from .SubLayers import MultiHeadAttention

__author__ = "Yu-Hsiang Huang"


class PositionWiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_2(F.relu(self.w_1(x)))
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, slf_attn_mask=None):
        attn_output, _ = self.mha(x, x, x, mask=slf_attn_mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(x + attn_output)

        ffn_output = self.pos_ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out1 = self.layer_norm2(out1 + ffn_output)

        return out1, []


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(x, x, x, mask=slf_attn_mask)
        attn1 = self.dropout1(dec_output)
        out1 = self.layer_norm1(attn1 + x)

        attn2, dec_attn_mask = self.enc_attn(out1, enc_output, enc_output, mask=dec_enc_attn_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layer_norm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layer_norm3(ffn_output + out2)

        return out3, [], []
