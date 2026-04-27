# Engram-Micro: A Replication Study at Sub-1B Scale

**Status:** Living draft. Each loop appends/edits.

---

## Abstract (placeholder)

DeepSeek's *Engram* (2026) inserts a hashed-N-gram conditional memory module
into a transformer to provide O(1) static-pattern lookup, freeing early
layers for compositional reasoning. Reported gains at 27B include +5.0 BBH
and +3.7 ARC-Challenge — i.e. the mechanism helps reasoning more than
knowledge, despite being motivated as a knowledge module. We ask whether
this mechanism still pays at sub-1B scale, where the backbone has fewer
"early layers" to free and where MoE is typically absent. We run a ρ-sweep
under two compute-matching protocols (iso-active and iso-param) on a small
GPT-style dense backbone trained on TinyStories, with synthetic probes for
the knowledge-vs-reasoning split. *Results pending.*

---

## 1 Introduction

Static patterns — named entities, formulaic phrases, multi-token canonical
forms — are reconstructed afresh at every transformer step. Engram replaces
this reconstruction with a hashed-N-gram lookup table, gated by current
context. The 27B-scale paper reports that the *biggest* gains are on
reasoning benchmarks, suggesting the mechanism's primary effect is making
the backbone effectively deeper, not just memorising more facts.

Whether this generalises below 1B parameters is open. The relevant scaling
variables — table coverage of the N-gram space, depth available to "free",
and the marginal value of FFN width vs. memory params — all change.

This work asks: **what is the optimal sparsity-allocation ratio ρ\* for a
small dense LM, and does the knowledge-vs-reasoning split survive?**

## 2 Related Work
- N-gram LMs (Kneser–Ney, etc.); hashed N-gram features in older NMT.
- Memory-augmented transformers: RETRO, kNN-LM, Memorizing Transformers.
- Small-LM corpus: SmolLM, TinyLlama, MobileLLM, OpenELM, OLMo-1B, Phi-1.5.
- *To-do: quote each, contrast Engram's static-vs-conditional gating.*

## 3 Architecture

### 3.1 Backbone
GPT-style pre-norm transformer with RMSNorm, RoPE, SwiGLU FFN, SDPA
attention, tied embeddings. Configurable depth and width.
(`engram_micro/model/lm.py`)

### 3.2 Engram module
Faithful single-stream variant of the reference demo
(`engram_micro/model/engram.py`):

- Tokenizer compression: case-fold + whitespace canonicalisation +
  Unicode normalisation; passed in as a `compressed_lookup` tensor at
  construction.
- Hashed N-gram retrieval: per-layer multipliers; per-head primes; depthwise
  causal short-conv with dilation = max_ngram_size to broaden temporal
  context without entangling adjacent retrievals.
- Multi-head: K independent retrievals concatenated and projected to
  hidden_size.
- Context-aware gate: `signed_sqrt(W_K(LN(h)))` modulates retrieved memory.

Deviations from the 27B implementation, each justified in
`.project/research/engram_spec.md`:
1. `hc_mult=1` (single stream): we study a dense backbone, not a
   hyper-connected multi-branch one.
2. `n_head_per_ngram=4` default (vs. 8): collisions less harmful at small
   table sizes.
3. End-to-end torch (vs. numpy hashing): GPU-resident, JIT-friendly.

### 3.3 Sparsity allocation at small scale

We define
```
P_active   = backbone (attn + FFN + norms) + Engram "compute" (W_K, W_V, conv)
P_table    = Engram embedding tables only
P_total    = P_active + P_table
ρ          = fraction of the sparse parameter budget in active backbone
```

`ρ=1` ↔ no Engram, full backbone.
`ρ=0` ↔ minimal backbone, full table.

See §4 for compute-matching protocol.

## 4 Compute Matching: Two Sweeps

(Summary of `.project/research/compute_matching.md`.)

We run two parallel sweeps:

- **Sweep A — iso-active**: backbone is identical for all ρ; table grows
  as ρ→0. P_active near-constant; P_total grows. Answers *given fixed
  per-token compute, does memory help?*
- **Sweep B — iso-param**: P_total held constant; FFN intermediate shrinks
  as table grows. P_active drops with ρ. Answers *given a fixed parameter
  budget, is FFN width or memory the better marginal spend?*

Reported jointly so neither confound (more total params; less FFN) goes
unattributed.

## 5 Evaluation

Three families:

- **PPL** on TinyStories validation. Floor measurement.
- **NameRecall**: synthetic story-template probe testing whether a
  named entity introduced once is correctly predicted on second
  reference. Static-pattern proxy; this is exactly what Engram should help.
- **InductionCopy**: random rare-token sequence with marker; measures
  the induction circuit. Reasoning-circuit proxy. Engram should at worst
  not hurt this, ideally help it (because early layers are freed).

Each probe reports top-1 accuracy, mean rank, and gold log-prob with
bootstrap 95% CIs.

## 6 Experiments

*Pending.* Section will be filled once Sweep A and Sweep B finish.

## 7 Mechanistic Analysis

*Pending.* Will replicate at small scale:
- Logit-lens accelerated-prediction-convergence test
- Per-token gate activation distribution + correlation with token "staticness"
- Attention-head entropy with vs. without Engram

## 8 Edge-deployment considerations

*Pending.*

## 9 Discussion

*Pending.*
