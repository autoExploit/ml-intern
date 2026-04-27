# Compute Matching at Small Scale — Decision

**Open issue from Loop 3.** The original Engram paper reports iso-parameter,
iso-FLOPs comparisons. At 27B, this is achieved by reducing the *number of
routed experts* in the MoE while keeping each expert's compute identical, so
P_active stays constant and the freed parameter budget moves to the Engram
table. The trade is clean: same per-token compute, different parameter
allocation between active backbone and (compute-free) memory lookup.

At our scale the backbone is *dense* (not MoE). Shrinking parameters means
shrinking FFN width or depth, and that *does* change compute. So a single
ρ-sweep cannot simultaneously hold P_active and P_total constant.

## Decision: run TWO sweeps, report both.

### Sweep A — iso-active (cleanest test of "memory helps fixed compute")
- Backbone arch is **identical** across all ρ values.
- Engram table size scales linearly: `P_table = (1-ρ) * T_max`.
- P_active varies only by Engram's small W_K/W_V/conv "compute" params
  (sub-1% of P_active in our configs).
- P_total grows as ρ→0 (more memory).
- **Question this answers:** *Given a fixed compute budget, does adding more
  static-pattern memory help, and how much?*
- Implemented in `accounting.py::design_iso_active_configs`.

### Sweep B — iso-param (cleanest test of "memory beats more FFN params")
- P_total held ~constant across ρ.
- At ρ=1: full-width FFN, no Engram. At ρ<1: FFN intermediate shrunk so
  freed params equal `(1-ρ) * T_max` Engram table.
- P_active *decreases* as ρ→0 (smaller FFN = less compute per token).
- **Question this answers:** *Given a fixed total parameter budget, is it
  better to spend marginal params on FFN width or on memory table?*
- Implemented in `accounting.py::design_iso_param_configs`.

### Why we need both
- Sweep A alone confounds "memory helps" with "more total params helps".
- Sweep B alone confounds "memory helps" with "less FFN compute hurts".
- Together: if Sweep A shows ρ*<1 AND Sweep B shows ρ*<1, that's evidence
  for memory-helps-fixed-compute AND memory-beats-FFN-width — robust.
- If Sweep A shows ρ*<1 but Sweep B shows ρ*=1 → memory helps only when it's
  free; trading FFN for memory loses. That's the "more compute wins" world.

### Compute control on top of this
Both sweeps fix:
- training tokens (same dataset, same total tokens consumed)
- sequence length, batch size
- optimizer hyperparameters
- random seeds (≥3 seeds per config when feasible)

Wall-clock differs between Sweep B configs (smaller FFN → faster). We accept
this — the controlled variable is *parameters and tokens*, not wall-clock.

### Reporting convention
Every results JSON includes `design_mode` and the full param breakdown
(P_active, P_table, P_total). Plots will be ρ-vs-{val_loss, NameRecall_acc,
InductionCopy_acc}, one panel per sweep, with the iso-{active,param}
shaded region noted. Knowledge and reasoning probe results plotted
separately so we can detect the knowledge-vs-reasoning split that's
Engram's central mechanistic claim.
