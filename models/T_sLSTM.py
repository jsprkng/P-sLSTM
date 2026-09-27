import math

import torch
import torch.nn as nn
from einops import rearrange

from .xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from .xlstm.blocks.slstm.block import sLSTMBlockConfig
from .xlstm.blocks.slstm.layer import sLSTMLayerConfig
from .xlstm.components.feedforward import FeedForwardConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positions, so the encoder can order the patches."""

    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        divisor = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * divisor)
        pe[:, 1::2] = torch.cos(position * divisor[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.shape[1], :]


class Model(nn.Module):
    """P-sLSTM with a Transformer encoder before the sLSTM stack.

    patches -> embedding -> + positions -> TransformerEncoder -> sLSTM stack -> flatten -> head

    Everything except the encoder and positions is identical to P_sLSTM, so a
    comparison between the two isolates the encoder's contribution. With
    t_layers=0 the encoder and positions are skipped and the model computes
    exactly what P_sLSTM does.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        if configs.t_layers < 0:
            raise ValueError('t_layers must be 0 or more, got {}'.format(configs.t_layers))
        if configs.t_layers > 0 and configs.embedding_dim % configs.t_heads:
            raise ValueError('embedding_dim={} must be divisible by t_heads={}'.format(
                configs.embedding_dim, configs.t_heads))
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channel = configs.channel
        self.embedding_dim = configs.embedding_dim
        self.patch_size = configs.patch_size  # patch size
        self.stride = configs.stride  # the stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1  # the number of patches

        cfg = xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(backend=configs.backend,
                                       num_heads=configs.num_heads,
                                       conv1d_kernel_size=configs.conv1d_kernel_size,
                                       bias_init="powerlaw_blockdependent"),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=self.seq_len,
            num_blocks=configs.num_blocks,
            embedding_dim=configs.embedding_dim,
            slstm_at="all"
        )

        self.embedding = nn.Linear(self.patch_size, self.embedding_dim)
        if configs.t_layers > 0:
            self.positional = PositionalEncoding(self.embedding_dim, max_len=max(self.patch_num, 16))
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.embedding_dim,
                    nhead=configs.t_heads,
                    dim_feedforward=4 * self.embedding_dim,
                    dropout=configs.dropout,
                    batch_first=True,
                ),
                num_layers=configs.t_layers,
            )
        else:
            self.positional = nn.Identity()
            self.encoder = nn.Identity()
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.projection = nn.Linear(self.embedding_dim * self.patch_num, self.pred_len)

    def forward(self, x):

        # x: [Batch, Input length, Channel]
        B, L, C = x.shape

        # Channel independence, then patches
        x = rearrange(x, 'b l c -> (b c) l')
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # [(B * C), Num_patches, Patch_size]

        x = self.embedding(x)  # [(B * C), Num_patches, Embedding_dim]

        x = self.encoder(self.positional(x))  # [(B * C), Num_patches, Embedding_dim]

        x = self.xlstm_stack(x)  # [(B * C), Num_patches, Embedding_dim]

        x = x.flatten(1)  # [(B * C), Num_patches * Embedding_dim]

        x = self.projection(x)  # [(B * C), Pred_len]

        x = rearrange(x, '(b c) l -> b l c', b=B, c=C)  # [B, Pred_len, C]

        return x
