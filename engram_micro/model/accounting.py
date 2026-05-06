"""Parameter accounting for Engram-Micro.

Defines our small-scale notion of ρ:

    P_active = backbone params (embed excl., LM head excl., norms incl.,
               attn + FFN) + Engram non-table params (W_K, W_V, conv, norms).
    P_table  = Engram embedding tables only.
    P_total  = P_active + P_table.

`ρ` is then defined relative to a reference baseline `P_active_ref`:

    ρ = 1                  -> all budget in active backbone (no Engram)
    ρ = 0                  -> all "spare" budget in tables, minimum-size backbone

Concretely, given a maximum-table size choice T_max and a choice of how
much FFN width to "give up" at ρ=0, we generate matched configs by
shrinking the FFN's intermediate size and reallocating params to the
Engram table while keeping P_total constant.

In small-scale we won't perfectly match P_active across all ρ (Engram's
W_K/W_V add a few hundred K params). We track the residual error.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

from .lm import EngramLMConfig, EngramLM, BackboneConfig
from .engram import EngramConfig


def _module_param_count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_params(model: EngramLM) -> Dict[str, int]:
    """Decompose params into named buckets.

    Buckets:
      embed       — input embeddings (excluded from P_active typically)
      lm_head     — output head (tied -> 0)
      backbone_attn
      backbone_ffn
      backbone_norms
      engram_table
      engram_compute (W_K, W_V, conv, gate-norms)
    """
    out = dict(embed=0, lm_head=0, backbone_attn=0, backbone_ffn=0,
               backbone_norms=0, engram_table=0, engram_compute=0)
    out["embed"] = model.tok_embed.weight.numel()
    if model._lm_head_module is not None:
        out["lm_head"] = model._lm_head_module.weight.numel()
    for blk in model.blocks:
        out["backbone_attn"] += _module_param_count(blk.attn)
        out["backbone_ffn"] += _module_param_count(blk.ffn)
        out["backbone_norms"] += blk.ln1.weight.numel() + blk.ln2.weight.numel()
        if blk.engram is not None:
            e = blk.engram
            out["engram_table"] += e.embedding.weight.numel()
            out["engram_compute"] += (
                _module_param_count(e.W_V) + _module_param_count(e.W_K)
                + e.q_norm.weight.numel() + e.k_norm.weight.numel()
            )
            if e.conv is not None:
                out["engram_compute"] += _module_param_count(e.conv) + e.conv_norm.weight.numel()
            out["backbone_norms"] += blk.ln_e.weight.numel()
    out["backbone_norms"] += model.ln_f.weight.numel()
    out["P_active"] = (out["backbone_attn"] + out["backbone_ffn"]
                       + out["backbone_norms"] + out["engram_compute"])
    out["P_total"] = out["P_active"] + out["engram_table"]
    out["P_total_with_embed"] = out["P_total"] + out["embed"] + out["lm_head"]
    return out


def design_iso_param_configs(
    base_backbone: BackboneConfig,
    base_engram: EngramConfig,
    rho: float,
    max_engram_table_params: int,
    engram_layer_ids,
) -> Tuple[EngramLMConfig, Dict[str, int]]:
    """ISO-PARAM sweep: hold P_total ~ constant; P_active varies.

    Strategy:
      - At rho=1: no Engram (or zero-sized table). Backbone uses its
        default ffn intermediate size i_ref.
      - At rho<1: shrink ffn intermediate by Δ_ffn so that the freed
        backbone params equal the table allocation (1-rho)*max_table.
        We approximate: freed_ffn_params = 3 * d * Δ_i * num_layers
                        engram_table_params = (1-rho) * max_engram_table_params
        Solve for Δ_i.

    Returns (config, accounting_estimate).
    """
    bb = BackboneConfig(**asdict(base_backbone))
    if rho >= 0.999:
        cfg = EngramLMConfig(backbone=bb, engram=None, engram_layer_ids=[])
        return cfg, {"target_table": 0, "ffn_shrink": 0}

    target_table_params = int((1 - rho) * max_engram_table_params)

    # configure engram table to hit ~target_table_params.
    # table params = sum_over_layers (N-1)*K * prime_avg * d_per_head
    n_engram_layers = len(engram_layer_ids)
    K = base_engram.n_head_per_ngram
    Nm1 = base_engram.max_ngram_size - 1
    d_ph = base_engram.d_per_head
    per_slot_per_head = d_ph
    total_slots_needed = target_table_params // per_slot_per_head
    slots_per_head = max(8, total_slots_needed // (n_engram_layers * Nm1 * K))
    eng = EngramConfig(**asdict(base_engram))
    eng.base_table_size = int(slots_per_head)

    # estimate freed ffn params we need
    freed_needed = target_table_params  # iso-param: backbone shrinks by exactly this

    # Each FFN has 3 matrices of (d × i). Shrinking i by Δ in every layer
    # gives 3*d*Δ*num_layers freed params.
    bb_ref = base_backbone
    i_ref = SwiGLU_intermediate(bb_ref.hidden_size, bb_ref.ffn_mult)
    delta_i = freed_needed / (3 * bb_ref.hidden_size * bb_ref.num_layers)
    delta_i = int(round(delta_i / 8) * 8)
    new_i = max(8, i_ref - delta_i)
    # convert back to ffn_mult that yields this i:
    # i = round(ffn_mult * d * 2/3) and rounded to multiple of 8.
    # Just store ffn_mult so SwiGLUFFN regenerates near new_i.
    bb.ffn_mult = (new_i * 3.0 / 2.0) / bb_ref.hidden_size

    cfg = EngramLMConfig(backbone=bb, engram=eng, engram_layer_ids=engram_layer_ids)
    return cfg, {
        "target_table_params": target_table_params,
        "slots_per_head": slots_per_head,
        "i_ref": i_ref,
        "new_i": new_i,
        "ffn_mult": bb.ffn_mult,
    }


def SwiGLU_intermediate(d: int, ffn_mult: float) -> int:
    i = int(round(ffn_mult * d * 2 / 3))
    return max(8, (i + 7) // 8 * 8)


def design_iso_active_configs(
    base_backbone: BackboneConfig,
    base_engram: EngramConfig,
    rho: float,
    max_engram_table_params: int,
    engram_layer_ids,
) -> Tuple[EngramLMConfig, Dict[str, int]]:
    """ISO-ACTIVE (= iso-FLOPs) sweep: backbone is identical across all ρ;
    only the Engram table SIZE varies as (1-ρ)*max_engram_table_params.

    P_active changes only by a small constant due to Engram's W_K/W_V/conv
    (Engram "compute" params). Per-token FLOPs are dominated by the
    backbone, so this sweep is the cleanest test of "given the same
    training compute, does adding more memory help?".

    P_total varies linearly with (1-ρ): up to +max_engram_table_params at ρ=0.

    Returns (config, info).
    """
    bb = BackboneConfig(**asdict(base_backbone))
    if rho >= 0.999:
        cfg = EngramLMConfig(backbone=bb, engram=None, engram_layer_ids=[])
        return cfg, {"target_table_params": 0, "slots_per_head": 0}

    target_table_params = int((1 - rho) * max_engram_table_params)
    n_engram_layers = max(1, len(engram_layer_ids))
    K = base_engram.n_head_per_ngram
    Nm1 = base_engram.max_ngram_size - 1
    d_ph = base_engram.d_per_head
    total_slots_needed = target_table_params // d_ph
    slots_per_head = max(8, total_slots_needed // (n_engram_layers * Nm1 * K))
    eng = EngramConfig(**asdict(base_engram))
    eng.base_table_size = int(slots_per_head)

    cfg = EngramLMConfig(backbone=bb, engram=eng, engram_layer_ids=engram_layer_ids)
    return cfg, {
        "target_table_params": target_table_params,
        "slots_per_head": slots_per_head,
    }
