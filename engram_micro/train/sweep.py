"""ρ-sweep orchestrator.

Iterates over a list of ρ values, trains a model for each, runs the eval
harness on each checkpoint, and writes per-ρ JSON results to
`.project/results/sweeps/<mode>/<run_name>__rho<rho>.json`.

WHY: every science loop after this one needs to dispatch sweeps. Doing it
once via a single, audited runner means: (1) sweep configs live in code,
(2) all results land in one canonical location, (3) we can re-run a
single ρ deterministically.

Compute matching: the runner calls into the existing
`design_iso_active_configs` / `design_iso_param_configs` from
`accounting.py`. See `.project/research/compute_matching.md` for the
rationale of running both modes.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import List, Optional

import torch

from engram_micro.train.train import TrainConfig, train as train_one
from engram_micro.eval.run_eval import run_eval


def sweep(
    rhos: List[float],
    design_mode: str,
    base_train_cfg: TrainConfig,
    sweep_name: str,
    results_root: str = ".project/results/sweeps",
    eval_ppl_iters: int = 20,
    eval_probe_examples: int = 100,
) -> List[dict]:
    """Run a ρ-sweep. Returns list of per-ρ result dicts."""
    out_dir_root = os.path.join(results_root, design_mode)
    os.makedirs(out_dir_root, exist_ok=True)
    summary = []
    t_sweep = time.time()
    for rho in rhos:
        t0 = time.time()
        cfg = TrainConfig(**asdict(base_train_cfg))
        cfg.rho = rho
        cfg.design_mode = design_mode
        cfg.run_name = f"{sweep_name}__rho{rho:.2f}"
        cfg.out_dir = os.path.join("out", "sweeps", design_mode, cfg.run_name)
        os.makedirs(cfg.out_dir, exist_ok=True)
        print(f"\n=== sweep[{design_mode}] rho={rho} -> {cfg.out_dir} ===", flush=True)
        train_one(cfg)
        ckpt = os.path.join(cfg.out_dir, "last.pt")
        eval_rep = run_eval(
            ckpt=ckpt,
            val_dataset=cfg.dataset_name,
            val_split=cfg.val_split,
            tokenizer_name=cfg.tokenizer_name,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            ppl_iters=eval_ppl_iters,
            probe_examples=eval_probe_examples,
        )
        with open(os.path.join(cfg.out_dir, "results.json")) as f:
            train_results = json.load(f)
        rec = {
            "rho": rho,
            "design_mode": design_mode,
            "run_name": cfg.run_name,
            "ckpt": ckpt,
            "wallclock_s": time.time() - t0,
            "param_counts": train_results["counts"],
            "final_val_loss": train_results["final_val_loss"],
            "eval": eval_rep,
        }
        with open(os.path.join(out_dir_root, cfg.run_name + ".json"), "w") as f:
            json.dump(rec, f, indent=2)
        summary.append(rec)
    summary_path = os.path.join(out_dir_root, f"{sweep_name}__summary.json")
    with open(summary_path, "w") as f:
        json.dump({"sweep_name": sweep_name, "design_mode": design_mode,
                   "wallclock_s": time.time() - t_sweep, "runs": summary}, f, indent=2)
    print(f"\n=== sweep[{design_mode}] done in {time.time()-t_sweep:.1f}s -> {summary_path} ===",
          flush=True)
    return summary


def parse_rhos(s: str) -> List[float]:
    return [float(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rhos", type=parse_rhos, default=[1.0, 0.5, 0.0])
    p.add_argument("--design_mode", choices=["iso_active", "iso_param"],
                   default="iso_active")
    p.add_argument("--sweep_name", default="default")
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--backbone_d", type=int, default=128)
    p.add_argument("--backbone_layers", type=int, default=4)
    p.add_argument("--backbone_heads", type=int, default=4)
    p.add_argument("--engram_layer_ids", type=lambda s: [int(x) for x in s.split(",")],
                   default=[1, 2])
    p.add_argument("--max_engram_table_params", type=int, default=2_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_iters", type=int, default=10)
    p.add_argument("--eval_ppl_iters", type=int, default=20)
    p.add_argument("--eval_probe_examples", type=int, default=100)
    p.add_argument("--results_root", default=".project/results/sweeps")
    args = p.parse_args()

    base = TrainConfig(
        max_steps=args.max_steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        backbone_d=args.backbone_d,
        backbone_layers=args.backbone_layers,
        backbone_heads=args.backbone_heads,
        engram_layer_ids=args.engram_layer_ids,
        max_engram_table_params=args.max_engram_table_params,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        eval_iters=args.eval_iters,
        eval_every=0,  # rely on final eval only for sweep speed
        log_every=50,
    )
    sweep(args.rhos, args.design_mode, base, args.sweep_name,
          results_root=args.results_root,
          eval_ppl_iters=args.eval_ppl_iters,
          eval_probe_examples=args.eval_probe_examples)


if __name__ == "__main__":
    main()
