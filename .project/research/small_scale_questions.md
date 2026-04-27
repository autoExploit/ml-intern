# Small-Scale Open Questions for Engram-Micro

These are concrete sub-questions whose answers determine our experimental plan. Each links back to a section of `engram_spec.md`.

## 1. Definition of ρ for a *dense* backbone (spec §F)

The paper defines ρ as the split of `P_sparse = P_tot - P_act` between MoE-inactive expert params and Engram embedding params. A dense backbone has `P_sparse = 0` (everything is active). We need to redefine.

**Proposed definition (commit to this):**

> Hold `P_active = P_backbone + P_engram_compute` constant (this is what training FLOPs scale with — embeddings retrieved per token are O(1) but their *params* are non-active).
> Hold `P_total = P_backbone + P_engram_compute + P_engram_table` constant.
> `ρ ∈ [0,1]`: `P_engram_table = (1-ρ) * (P_total - P_active_baseline)`,
> with the freed params going into FFN width to keep `P_active` constant when `ρ→1`.

Concretely: pick a *baseline dense backbone* with `P_active_baseline = P_b`. For ρ<1 we *shrink* the FFN of that backbone by `Δ = (1-ρ) * Δ_max` (where `Δ_max` is the table size at ρ=0) and add a table of the same size `Δ`. Total params and per-token FLOPs are then *almost* matched (Engram compute params are tiny but not zero — we'll absorb them into the bookkeeping).

Open: what is `Δ_max`? Pick e.g. 80M for a ~120M backbone (so total 200M when ρ=0).

## 2. Layer placement at low depth

27B has Engram at layers 2 & 15 of 30 (≈7% & 50%). At depth 16:
- Proportional → layers 1 & 8.
- "Very early" hypothesis → layers 0 & 1, or 1 & 2.

Run with `[1,8]` then ablate to `[0,1]` and `[1, depth-2]` to test the early-relief hypothesis at small depth.

## 3. Number of hash heads K

Demo uses K=8. Smaller tables → higher collision rate with same K. But K dominates `d_mem = (N-1)*K*d_per_head`. For our budget keep `d_per_head=64` constant; ablate K∈{2,4,8}.

## 4. Tokenizer compression value at small vocab

Paper gets 23% reduction at 128k. With a 32k tokenizer the gain might be larger or smaller. **Mandatory ablation**: train one model with compression, one without, all else equal. If the gap < 0.5% of perplexity, drop compression (it's complexity).

## 5. Signed-sqrt gate trick

The demo code applies `gate = sign(x) * sqrt(|x|)` before sigmoid. Not in paper. **Mandatory ablation**: with vs without. If it doesn't matter, simpler gate (paper-faithful) wins.

## 6. Multi-branch (hyper-connections) interaction

Paper uses hyper-connections (M=4) and Engram is integrated into them. We deliberately *don't* use hyper-connections (single residual). This is a deviation. To validate: if the gain disappears in single-stream, hyper-connections were doing more work than the paper credits.

## 7. Conv vs no-conv

The depthwise causal conv (kernel=4, dilation=N) feels like a way to broaden the receptive field of retrieved embeddings. **Ablation**: disable the conv. If perf unchanged, drop it for simplicity.

## 8. Table init

Demo uses default `nn.Embedding` init (Normal mean=0, std=1). For sparse retrieval where the gate scales the embedding by σ(...) ∈ (0,1), maybe smaller init is better (avoid swamping the residual at start). **Ablation**: init scale ∈ {1.0, 0.1, 0.02}.

## 9. The core experiment design

Sparsity-allocation sweep over ρ ∈ {0, 0.2, 0.4, 0.6, 0.74, 0.8, 1.0}. All iso-`P_total` and approximately iso-`P_active`. Train for fixed token budget (start with 2-5B tokens at 100-200M scale; this is a feasibility constraint we'll scope when we cost it out).

Predicted curve: U-shape with optimum somewhere. Whether it's at the same ρ as 27B is the point.

## 10. Benchmarks we need

For perplexity: held-out validation slice of the training corpus.
For knowledge: a small closed-book QA / cloze task that should benefit from memorization. `lambada_openai`, `tinyMMLU`, or our own bigram-knowledge probe.
For reasoning: arithmetic / logic task. `arc_easy`, or a controlled synthetic reasoning task (chains of variable assignments).

We'll start with PPL + LAMBADA + ARC-Easy and possibly a synthetic "named-entity recall" probe to directly test the "static reconstruction" hypothesis.
