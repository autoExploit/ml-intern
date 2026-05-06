"""Run all evals on a saved checkpoint.

Usage:
    python -m engram_micro.eval.run_eval --ckpt out/run/last.pt \
        --val_dataset roneneldan/TinyStories --val_split validation

Writes a JSON report to <ckpt_dir>/eval.json with PPL + probe results.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import torch

from engram_micro.model.lm import EngramLM, EngramLMConfig, BackboneConfig
from engram_micro.model.engram import EngramConfig, build_compression_table
from engram_micro.data.loader import make_loader
from engram_micro.eval.perplexity import perplexity
from engram_micro.eval.probes import name_recall_probe, induction_copy_probe


def load_model(ckpt_path: str, tokenizer, device: str):
    """Reconstruct an EngramLM from a checkpoint dict produced by train.train."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    lm_cfg_dict = state["lm_config"]
    bb = BackboneConfig(**lm_cfg_dict["backbone"])
    eng_dict = lm_cfg_dict.get("engram")
    eng = EngramConfig(**eng_dict) if eng_dict else None
    cfg = EngramLMConfig(backbone=bb, engram=eng,
                         engram_layer_ids=lm_cfg_dict.get("engram_layer_ids", []))
    ctab = build_compression_table(tokenizer) if eng is not None else None
    model = EngramLM(cfg, compression_table=ctab)
    model.load_state_dict(state["model"], strict=False)
    return model.to(device), cfg


def run_eval(ckpt: str,
             val_dataset: str = "roneneldan/TinyStories",
             val_split: str = "validation",
             tokenizer_name: str = "gpt2",
             seq_len: int = 256,
             batch_size: int = 8,
             ppl_iters: int = 50,
             probe_examples: int = 200,
             device: "Optional[str]" = None) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, cfg = load_model(ckpt, tokenizer, device)

    val_iter = make_loader(val_dataset, val_split, tokenizer,
                           seq_len=seq_len, batch_size=batch_size,
                           text_field="text", seed=0)
    ppl = perplexity(model, val_iter, ppl_iters, device)

    nr = name_recall_probe(model, tokenizer, device, max_examples=probe_examples)
    ic = induction_copy_probe(model, tokenizer, device,
                              n_examples=probe_examples, seq_len=64)

    report = {
        "ckpt": ckpt,
        "ppl": ppl,
        "name_recall": asdict(nr),
        "induction_copy": asdict(ic),
    }
    out = os.path.join(os.path.dirname(ckpt), "eval.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    return report


# late import compatibility
from typing import Optional  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val_dataset", default="roneneldan/TinyStories")
    p.add_argument("--val_split", default="validation")
    p.add_argument("--tokenizer_name", default="gpt2")
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--ppl_iters", type=int, default=50)
    p.add_argument("--probe_examples", type=int, default=200)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    rep = run_eval(**vars(args))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
