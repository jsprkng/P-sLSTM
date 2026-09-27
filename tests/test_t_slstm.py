"""Checks for T_sLSTM, the Transformer + sLSTM variant of P-sLSTM.

    .venv/Scripts/python -m pytest tests -q
"""

import types

import pytest
import torch

from models import P_sLSTM, T_sLSTM

# Weather settings from scripts/EXP-LongForecasting/P_sLSTM/weather.sh
WEATHER = dict(seq_len=336, pred_len=96, channel=21, embedding_dim=100, patch_size=56, stride=56,
               num_heads=2, conv1d_kernel_size=8, num_blocks=2, dropout=0.1, backend='vanilla',
               t_layers=2, t_heads=4)


def configs(**overrides):
    return types.SimpleNamespace(**{**WEATHER, **overrides})


def batch(cfg, size=4):
    torch.manual_seed(0)
    return torch.randn(size, cfg.seq_len, cfg.channel)


def test_output_shape():
    cfg = configs()
    assert T_sLSTM.Model(cfg)(batch(cfg)).shape == (4, cfg.pred_len, cfg.channel)


@pytest.mark.parametrize('overrides', [
    dict(seq_len=336, patch_size=6, stride=6, channel=7, num_heads=4, conv1d_kernel_size=4),  # ETTm1
    dict(seq_len=96, patch_size=16, stride=16, channel=5, embedding_dim=200, conv1d_kernel_size=8),  # PEMS03, fewer channels
    dict(patch_size=16, stride=8),  # overlapping patches
])
def test_output_shape_across_paper_settings(overrides):
    cfg = configs(**overrides)
    assert T_sLSTM.Model(cfg)(batch(cfg, 2)).shape == (2, cfg.pred_len, cfg.channel)


def test_zero_encoder_layers_reproduces_p_slstm():
    cfg = configs(t_layers=0)
    p = P_sLSTM.Model(cfg).eval()
    t = T_sLSTM.Model(cfg).eval()
    t.load_state_dict(p.state_dict())  # strict: the two must have identical parameters
    x = batch(cfg)
    with torch.no_grad():
        assert torch.equal(t(x), p(x))


def test_encoder_changes_the_forecast():
    cfg = configs()
    t = T_sLSTM.Model(cfg).eval()
    p = P_sLSTM.Model(cfg).eval()
    t.load_state_dict(p.state_dict(), strict=False)  # shared parts equal, encoder on top
    x = batch(cfg)
    with torch.no_grad():
        assert not torch.allclose(t(x), p(x))


def test_encoder_receives_gradients():
    cfg = configs()
    t = T_sLSTM.Model(cfg)
    t(batch(cfg)).pow(2).mean().backward()
    for name, param in t.encoder.named_parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0, name


def test_channels_are_processed_independently():
    cfg = configs()
    t = T_sLSTM.Model(cfg).eval()
    x = batch(cfg)
    perm = torch.randperm(cfg.channel)
    with torch.no_grad():
        assert torch.allclose(t(x)[:, :, perm], t(x[:, :, perm]), atol=1e-5)


def test_heads_must_divide_embedding_dim():
    with pytest.raises(ValueError, match='divisible'):
        T_sLSTM.Model(configs(t_heads=3))


def test_negative_encoder_layers_rejected():
    with pytest.raises(ValueError, match='t_layers'):
        T_sLSTM.Model(configs(t_layers=-1))
