import math
import torch
from torch import nn, Tensor
from .Layers import EncoderLayer, DecoderLayer

__author__ = "oak"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).to(torch.float32)
        div_term = torch.exp(torch.arange(0., d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin((position * div_term))
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Encoder(nn.Module):

    def __init__(
            self, n_layers, n_head, d_k, d_v, d_model, dff, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.src_word_emb = nn.Linear(dff, d_model, bias=False)

        # n_layers Encoder layer.
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, dff, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)]
        )

    def forward(self, x, src_mask):
        """  """
        x = self.src_word_emb(x)
        x = self.dropout(self.position_enc(x))
        x = self.layer_norm(x)

        for enc_layer in self.enc_layers:
            x, _ = enc_layer(x, slf_attn_mask=src_mask)

        return x, []


class Decoder(nn.Module):

    def __init__(self, n_trg_vocab, n_layers, n_head, d_k, d_v, d_model, dff, pad_idx, dropout=0.1):

        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(
            n_trg_vocab, d_model, padding_idx=pad_idx
        )

        self.dropout1 = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, dff, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)
        ])

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attn=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        x = self.embedding(trg_seq)
        x *= self.d_model ** 0.5  # scaling
        x = self.dropout1(self.position_enc(x))
        x = self.layer_norm1(x)

        for dec_layer in self.dec_layers:
            x, dec_slf_attn, dec_enc_attn = dec_layer(
                x, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=None)
            dec_slf_attn_list += [dec_slf_attn] if return_attn else []
            dec_enc_attn_list += [dec_enc_attn] if return_attn else []

        if return_attn:
            return x, dec_slf_attn_list, dec_enc_attn_list
        return x, []


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, n_trg_vocab, trg_pad_idx, d_model=512, dff=2048, n_layers=4, n_head=8, d_k=64, d_v=64,
                 dropout=0.1, trg_emb_prj_weight_sharing=True):

        super().__init__()

        self.scale_prj = True
        self.d_model = d_model
        self.trg_pad_idx = trg_pad_idx

        self.encoder = Encoder(d_model=d_model, dff=dff, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                               dropout=dropout)

        self.decoder = Decoder(n_trg_vocab=n_trg_vocab, d_model=d_model, dff=dff, n_layers=n_layers, n_head=n_head,
                               d_k=d_k, d_v=d_v, pad_idx=trg_pad_idx, dropout=dropout)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        # if trg_emb_prj_weight_sharing:
        #     # Share the weight between target word embedding & last dense layer
        #     self.trg_word_prj.weight = self.decoder.embedding.weight

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inp, trg_seq):

        trg_mask = get_pad_mask(
            trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
        trg_mask = trg_mask.view(
            trg_mask.shape[0], -1, trg_mask.shape[1], trg_mask.shape[2])

        enc_output, *_ = self.encoder(inp, None)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, trg_mask)
        seq_output = self.trg_word_prj(dec_output)

        # if self.scale_prj:
        #     seq_output *= self.d_model ** -0.5

        return seq_output
