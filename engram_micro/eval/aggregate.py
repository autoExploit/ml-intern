"""Aggregate and plot sweep results.

Reads all per-rho JSONs under `.project/results/sweeps/<mode>/` and
produces (a) a summary CSV and (b) a matplotlib figure with three panels:
val_loss, NameRecall mean_rank, InductionCopy mean_rank, all vs ρ.

We plot mean_rank rather than top-1 because at small scale top-1 stays
near 0 for both probes; rank is the sensitive metric.

Output:
    <results_root>/<mode>/_summary.csv
    <results_root>/<mode>/_summary.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import List


def load_sweep(mode_dir: str) -> List[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(mode_dir, "*.json"))):
        if path.endswith("_summary.json") or os.path.basename(path).startswith("_"):
            continue
        with open(path) as f:
            runs.append(json.load(f))
    runs.sort(key=lambda r: r["rho"])
    return runs


def write_csv(runs: List[dict], path: str):
    fields = ["rho", "design_mode", "P_active", "P_total", "final_val_loss",
              "ppl", "name_recall_top1", "name_recall_rank", "name_recall_logp",
              "induction_top1", "induction_rank", "induction_logp",
              "wallclock_s"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in runs:
            ev = r["eval"]
            w.writerow([
                r["rho"], r["design_mode"],
                r["param_counts"]["P_active"], r["param_counts"]["P_total"],
                r["final_val_loss"],
                ev["ppl"]["ppl"],
                ev["name_recall"]["top1_acc"], ev["name_recall"]["mean_rank"],
                ev["name_recall"]["mean_logprob_gold"],
                ev["induction_copy"]["top1_acc"], ev["induction_copy"]["mean_rank"],
                ev["induction_copy"]["mean_logprob_gold"],
                r["wallclock_s"],
            ])


def make_plot(runs: List[dict], path: str, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available, skipping figure")
        return
    rhos = [r["rho"] for r in runs]
    vloss = [r["final_val_loss"] for r in runs]
    nr_rank = [r["eval"]["name_recall"]["mean_rank"] for r in runs]
    ic_rank = [r["eval"]["induction_copy"]["mean_rank"] for r in runs]
    nr_lp = [r["eval"]["name_recall"]["mean_logprob_gold"] for r in runs]
    ic_lp = [r["eval"]["induction_copy"]["mean_logprob_gold"] for r in runs]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(rhos, vloss, marker="o"); axes[0, 0].set_title("val loss")
    axes[0, 0].set_xlabel("ρ"); axes[0, 0].invert_xaxis()
    axes[0, 1].plot(rhos, nr_rank, marker="o", color="C1")
    axes[0, 1].set_title("NameRecall mean rank (lower=better)")
    axes[0, 1].set_xlabel("ρ"); axes[0, 1].invert_xaxis()
    axes[1, 0].plot(rhos, nr_lp, marker="o", color="C2")
    axes[1, 0].set_title("NameRecall mean log-prob of gold (higher=better)")
    axes[1, 0].set_xlabel("ρ"); axes[1, 0].invert_xaxis()
    axes[1, 1].plot(rhos, ic_lp, marker="o", color="C3")
    axes[1, 1].set_title("InductionCopy mean log-prob of gold")
    axes[1, 1].set_xlabel("ρ"); axes[1, 1].invert_xaxis()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default=".project/results/sweeps")
    p.add_argument("--mode", choices=["iso_active", "iso_param"], required=True)
    args = p.parse_args()
    mode_dir = os.path.join(args.results_root, args.mode)
    runs = load_sweep(mode_dir)
    if not runs:
        print(f"no runs in {mode_dir}")
        return
    write_csv(runs, os.path.join(mode_dir, "_summary.csv"))
    make_plot(runs, os.path.join(mode_dir, "_summary.png"),
              title=f"Engram-Micro ρ-sweep ({args.mode})")
    print(f"wrote {mode_dir}/_summary.{{csv,png}}")
    for r in runs:
        ev = r["eval"]
        print(f"  ρ={r['rho']:.2f}  vloss={r['final_val_loss']:.3f}  "
              f"PPL={ev['ppl']['ppl']:.1f}  "
              f"NR_rank={ev['name_recall']['mean_rank']:.0f}  "
              f"IC_rank={ev['induction_copy']['mean_rank']:.0f}")


if __name__ == "__main__":
    main()
