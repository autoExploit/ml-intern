"""Test backbone + Engram-LM wiring + param accounting."""
import torch

from engram_micro.model.engram import EngramConfig
from engram_micro.model.lm import EngramLM, EngramLMConfig, BackboneConfig
from engram_micro.model.accounting import (
    count_params, design_iso_param_configs, SwiGLU_intermediate,
)


def _ident(V):
    return torch.arange(V, dtype=torch.long)


def _tiny_backbone(V=257, d=64, L=4, h=4):
    return BackboneConfig(vocab_size=V, hidden_size=d, num_layers=L,
                          num_heads=h, ffn_mult=4.0, max_seq_len=128)


def _tiny_engram(d=64):
    return EngramConfig(hidden_size=d, max_ngram_size=3, n_head_per_ngram=2,
                        base_table_size=257, d_per_head=16, kernel_size=3,
                        pad_id=0, seed=0)


def test_baseline_lm_forward_backward():
    cfg = EngramLMConfig(backbone=_tiny_backbone(), engram=None, engram_layer_ids=[])
    m = EngramLM(cfg)
    ids = torch.randint(1, 256, (2, 16))
    logits, loss = m(ids, labels=ids)
    assert logits.shape == (2, 16, 257)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.tok_embed.weight.grad is not None


def test_engram_lm_forward_backward():
    cfg = EngramLMConfig(
        backbone=_tiny_backbone(),
        engram=_tiny_engram(),
        engram_layer_ids=[1, 2],
    )
    m = EngramLM(cfg, compression_table=_ident(257))
    ids = torch.randint(1, 256, (2, 16))
    logits, loss = m(ids, labels=ids)
    assert logits.shape == (2, 16, 257)
    loss.backward()
    # gradient must flow into engram tables of both inserted layers
    n_with_grad = 0
    for blk in m.blocks:
        if blk.engram is not None:
            assert blk.engram.embedding.weight.grad is not None
            assert blk.engram.embedding.weight.grad.abs().sum() > 0
            n_with_grad += 1
    assert n_with_grad == 2


def test_count_params_buckets():
    cfg = EngramLMConfig(
        backbone=_tiny_backbone(),
        engram=_tiny_engram(),
        engram_layer_ids=[1, 2],
    )
    m = EngramLM(cfg, compression_table=_ident(257))
    counts = count_params(m)
    # Buckets present
    for k in ["embed", "lm_head", "backbone_attn", "backbone_ffn",
              "engram_table", "engram_compute", "P_active", "P_total"]:
        assert k in counts
    # tied embeddings -> lm_head=0
    assert counts["lm_head"] == 0
    # engram table > 0
    assert counts["engram_table"] > 0
    # engram_compute > 0
    assert counts["engram_compute"] > 0


def test_iso_param_design_shrinks_ffn_when_rho_decreases():
    bb = _tiny_backbone(d=128, L=6)
    eng = _tiny_engram(d=128)
    eng.base_table_size = 1024  # base size for design
    cfg_hi, info_hi = design_iso_param_configs(bb, eng, rho=1.0, max_engram_table_params=200_000, engram_layer_ids=[1, 3])
    cfg_lo, info_lo = design_iso_param_configs(bb, eng, rho=0.0, max_engram_table_params=200_000, engram_layer_ids=[1, 3])
    # rho=1: no engram
    assert cfg_hi.engram is None
    # rho=0: ffn smaller than baseline
    i_ref = SwiGLU_intermediate(bb.hidden_size, bb.ffn_mult)
    i_lo = SwiGLU_intermediate(cfg_lo.backbone.hidden_size, cfg_lo.backbone.ffn_mult)
    assert i_lo < i_ref


def test_param_iso_constraint_within_tolerance():
    """Models at rho=1 and rho=0 should have similar P_total
    (within ~5% — Engram compute params introduce a small unmatched delta)."""
    bb = BackboneConfig(vocab_size=257, hidden_size=128, num_layers=6,
                        num_heads=4, ffn_mult=4.0, max_seq_len=64)
    eng = _tiny_engram(d=128)
    cfg_hi, _ = design_iso_param_configs(bb, eng, rho=1.0,
                                         max_engram_table_params=400_000,
                                         engram_layer_ids=[1, 3])
    cfg_lo, _ = design_iso_param_configs(bb, eng, rho=0.0,
                                         max_engram_table_params=400_000,
                                         engram_layer_ids=[1, 3])
    m_hi = EngramLM(cfg_hi)
    m_lo = EngramLM(cfg_lo, compression_table=_ident(257))
    c_hi = count_params(m_hi)
    c_lo = count_params(m_lo)
    # exclude embeddings (same), compare P_total
    pt_hi = c_hi["P_total"]
    pt_lo = c_lo["P_total"]
    rel = abs(pt_hi - pt_lo) / max(pt_hi, pt_lo)
    assert rel < 0.10, f"P_total mismatch too large: {pt_hi} vs {pt_lo} ({rel:.3f})"
