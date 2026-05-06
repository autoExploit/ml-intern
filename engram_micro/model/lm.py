"""Minimal GPT-style dense transformer for Engram-Micro.

Design:
- Pre-norm RMSNorm.
- Multi-head attention via torch's scaled_dot_product_attention (causal=True).
- SwiGLU FFN (closer to modern small-LM design — TinyLlama, SmolLM).
- Tied input/output embeddings (saves params and standard at this scale).
- Rotary positional embedding (RoPE), simple inline implementation.

Engram integration:
- Inserted as a *parallel residual* sub-block at layers in cfg.engram_layer_ids.
- Order in those layers:  H ← H + Engram(H, ids); H ← H + Attn(LN(H)); H ← H + FFN(LN(H)).
  This mirrors the demo's TransformerBlock ordering.

We deliberately keep this small and grokkable. No KV cache (training only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .engram import EngramConfig, EngramMemory


# ---------------------------------------------------------------------------
# Backbone config
# ---------------------------------------------------------------------------

@dataclass
class BackboneConfig:
    vocab_size: int = 50257                 # GPT-2 BPE default
    hidden_size: int = 512
    num_layers: int = 12
    num_heads: int = 8
    ffn_mult: float = 4.0                   # SwiGLU intermediate ≈ 8/3 * hidden by convention
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tied_embeddings: bool = True


# ---------------------------------------------------------------------------
# Rotary position embedding (RoPE)
# ---------------------------------------------------------------------------

def _rope_freqs(head_dim: int, seq_len: int, theta: float, device, dtype):
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    angles = torch.outer(t, freqs)  # [L, half]
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def _apply_rope(x: torch.Tensor, cos, sin) -> torch.Tensor:
    # x: [B, H, L, D]; D even.
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    cos_full = torch.cat([cos_b, cos_b], dim=-1)
    sin_full = torch.cat([sin_b, sin_b], dim=-1)
    return x * cos_full + rotated * sin_full


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        assert cfg.hidden_size % cfg.num_heads == 0
        self.h = cfg.num_heads
        self.d = cfg.hidden_size // cfg.num_heads
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.o = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        # SwiGLU: hidden -> 2*intermediate (gate, up) -> hidden
        # Intermediate sized so (gate+up+down) ≈ ffn_mult * d^2 * 3 params.
        # We pick intermediate so total params ≈ ffn_mult * d * d.
        # Three matrices of shape (d -> i): so total = 3*d*i. Set i = ffn_mult * d / 3 * (something).
        # Simpler: just specify intermediate = round(ffn_mult * d * 2/3) (Llama convention).
        i = int(round(cfg.ffn_mult * cfg.hidden_size * 2 / 3))
        # Round to nearest multiple of 8 for hardware friendliness.
        i = max(8, (i + 7) // 8 * 8)
        self.intermediate_size = i
        self.w_gate = nn.Linear(cfg.hidden_size, i, bias=False)
        self.w_up = nn.Linear(cfg.hidden_size, i, bias=False)
        self.w_down = nn.Linear(i, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        cfg: BackboneConfig,
        layer_id: int,
        engram: Optional[EngramMemory] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.engram = engram
        self.attn = CausalSelfAttention(cfg)
        self.ffn = SwiGLUFFN(cfg)
        self.ln1 = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.ln2 = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        if engram is not None:
            self.ln_e = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)

    def forward(self, h, input_ids, cos, sin):
        if self.engram is not None:
            h = h + self.engram(self.ln_e(h), input_ids)
        h = h + self.attn(self.ln1(h), cos, sin)
        h = h + self.ffn(self.ln2(h))
        return h


# ---------------------------------------------------------------------------
# Full LM
# ---------------------------------------------------------------------------

@dataclass
class EngramLMConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    engram: Optional[EngramConfig] = None     # if None -> pure baseline
    engram_layer_ids: List[int] = field(default_factory=lambda: [1, 6])


class EngramLM(nn.Module):
    """Transformer LM with optional Engram modules at chosen layers."""

    def __init__(
        self,
        cfg: EngramLMConfig,
        compression_table: Optional[torch.LongTensor] = None,
    ):
        super().__init__()
        self.cfg = cfg
        bb = cfg.backbone
        self.tok_embed = nn.Embedding(bb.vocab_size, bb.hidden_size)
        nn.init.normal_(self.tok_embed.weight, std=0.02)

        engram_layers = set(cfg.engram_layer_ids) if cfg.engram is not None else set()
        if cfg.engram is not None:
            assert compression_table is not None, "Engram requires a compression_table"
            # ensure engram cfg is consistent with backbone
            cfg.engram.hidden_size = bb.hidden_size

        blocks = []
        for li in range(bb.num_layers):
            engram = None
            if li in engram_layers:
                engram = EngramMemory(cfg.engram, compression_table, layer_id=li)
            blocks.append(TransformerBlock(bb, layer_id=li, engram=engram))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.RMSNorm(bb.hidden_size, eps=bb.norm_eps)

        if bb.tied_embeddings:
            self.lm_head = lambda x: x @ self.tok_embed.weight.T
            self._lm_head_module = None
        else:
            self._lm_head_module = nn.Linear(bb.hidden_size, bb.vocab_size, bias=False)
            nn.init.normal_(self._lm_head_module.weight, std=0.02)
            self.lm_head = self._lm_head_module

        # init linear weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # cache RoPE for max_seq_len
        self.register_buffer("_rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("_rope_sin", torch.empty(0), persistent=False)

    def _rope(self, L, device, dtype):
        if self._rope_cos.numel() < L * (self.cfg.backbone.hidden_size // self.cfg.backbone.num_heads // 2):
            head_dim = self.cfg.backbone.hidden_size // self.cfg.backbone.num_heads
            cos, sin = _rope_freqs(head_dim, max(L, self.cfg.backbone.max_seq_len),
                                   self.cfg.backbone.rope_theta, device, dtype)
            self._rope_cos = cos
            self._rope_sin = sin
        return self._rope_cos[:L], self._rope_sin[:L]

    def forward(self, input_ids: torch.LongTensor,
                labels: Optional[torch.LongTensor] = None):
        B, L = input_ids.shape
        h = self.tok_embed(input_ids)
        cos, sin = self._rope(L, h.device, h.dtype)
        for blk in self.blocks:
            h = blk(h, input_ids, cos, sin)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            # next-token prediction: predict labels[..., 1:] from logits[..., :-1, :]
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss
