# Engram-Micro Loop Log

## Loop 0 — Bootstrap

### OBSERVE
- Repo `ml-intern` exists with unrelated agent scaffolding (backend/frontend/agent dirs, HF Pro tooling). Not relevant to Engram-Micro mission. Will treat root as workspace; new code will go under `engram_micro/`.
- Remote: `https://github.com/autoExploit/ml-intern` configured.
- No `.project/` dir existed → created `research/`, `report/`, `results/`.
- No prior Engram code, no research notes, no results.

### ORIENT
- Highest leverage right now: **understand Engram precisely** before writing a single line. Briefing in the prompt is detailed but must be verified vs the paper + repo. If the briefing is wrong on a key detail (e.g., what the "memory table" actually stores per slot, how multi-head hashing partitions are organized, what gating signal is), every downstream decision compounds error.
- Not yet meaningful to refactor or pick a baseline backbone — those decisions depend on memory-table mechanics.

### DECIDE
**Action:** Research the Engram paper (arXiv 2601.07372) and the deepseek-ai/Engram GitHub repo. Produce `.project/research/engram_spec.md` capturing the *exact* mechanics of (1) tokenizer compression, (2) hashed N-gram lookup table shape & per-slot contents, (3) gating, (4) layer integration, (5) sparsity allocation. Also a short companion file `.project/research/small_scale_questions.md` listing what we expect changes at <1B.

**Falsifiable prediction:** After this loop I should be able to answer, with citations to specific files/equations: "what is the dtype, shape, and update rule of one Engram memory table entry?" If I can't, the research is incomplete.

### DEVIL'S ADVOCATE
- *Technical attack:* Paper might describe the table differently from the code (papers often lag). Mitigation: read code as ground truth, paper as motivation.
- *Experimental attack:* Just reading docs doesn't reduce risk unless I write down the *small-scale-relevant* invariants. Mitigation: explicit `small_scale_questions.md`.
- *Priority attack:* Could I skip research and just clone the official repo + scale it down? No — the mission is to understand mechanism well enough to deviate justifiably. Cloning would hide the assumptions that break at <1B.

Rebuttals stand. Proceeding.

### DO
(spawn research sub-agent — large, parallel-ish, benefits from isolated context: reading paper + repo end-to-end is exactly the use case)


---

## Loop 1 — Engram module

### OBSERVE
Loop 0 produced spec + small-scale questions. No code yet. Demo at `/tmp/Engram` reviewed.

### ORIENT / DECIDE
Highest leverage: implement the Engram module faithfully + unit-test, before backbone or training. Engram is the load-bearing novel piece.

### DEVIL'S ADVOCATE
- Tech: hash formula must match demo exactly. Used same `mix * mults[0] XOR mix * mults[k]` per-layer multipliers. Per-head primes unique globally for the layer.
- Tech: signed-sqrt gate squashing was not in paper, only in demo code. Reproduced under flag.
- Tech: conv dilation must = max_ngram_size (not 1).
- Exp: unit tests don't prove the module helps — only that it runs correctly. That's the right scope here.
- Priority: tokenizer choice deferred (module is tokenizer-agnostic; takes id arrays).

### DO
- Created package `engram_micro/{model,data,train,eval,tests}`.
- `model/engram.py`: 280-line implementation with EngramConfig, EngramMemory, build_compression_table.
- 12 unit tests covering: shape, determinism, gradient flow, pad handling, hash range, hash multiplier per-layer uniqueness, ablation flags (signed-sqrt, no-conv), causality (changing token at t leaves outputs at <t intact), compression collapses 'A'/'a' on real GPT-2 tokenizer, hash coverage > 50%.

### RESULT
All 12 tests pass on CPU torch 2.10. Build green.

### NEXT
- Backbone GPT-style transformer scaffold.
- Param-counting helper that lets us actually compute ρ for our redefined small-scale formulation.
- Tokenizer commitment (start with GPT-2 BPE; trivial to swap).

---

## Loop 2 — Backbone, integration, param accounting

### OBSERVE / ORIENT
Engram module green from Loop 1. Need backbone to host it + matched-config generator for the ρ sweep.

### DECIDE
Build (a) minimal GPT-style backbone (RMSNorm pre-norm, RoPE, SwiGLU FFN, SDPA causal attn, tied embed), (b) EngramLM that inserts EngramMemory at chosen layers, (c) param accounting + iso-param config designer.

Falsifiable: end-to-end fwd/bwd works for both baseline + engram; iso-param sweep configs come within 5% of constant P_total across ρ∈[0,1].

### DEVIL'S ADVOCATE
- Tech: SDPA gives causal mask for free; safer than handrolled.
- Tech: SwiGLU intermediate computed as `round(ffn_mult*d*2/3)` rounded to multiple of 8 (Llama convention).
- Exp: this loop doesn't train anything, so doesn't validate Engram helps. Right scope.
- Priority: param accounting was the real upstream — without it ρ-sweep is impossible.

### DO
- `model/lm.py`: BackboneConfig, RoPE helpers, CausalSelfAttention (SDPA, RoPE), SwiGLUFFN, TransformerBlock (with optional Engram), EngramLM (forward+loss).
- `model/accounting.py`: count_params() bucketing all params; design_iso_param_configs(rho) that shrinks FFN intermediate to free space for the table.
- 5 new tests (17 total). All green.

### RESULT
At ~60M P_total target with backbone d=640, L=12, H=10, max_table_params=36M, ρ-sweep configs hit P_total within 2.5% (60.43M ↔ 60.56M). With embeddings, ~92.6M total each. P_active varies 24.5M→59.1M as ρ goes 0→1 — this means *iso-FLOPs is NOT held*. Iso-param yes; iso-FLOPs no.

This is a genuine scientific issue: at small scale, FFN params dominate compute, so different ρ → different training FLOPs. Two options:
(a) Stick with iso-param only; add a *separate* iso-active sweep where we hold P_active constant by adding *depth* instead of width when ρ→0.
(b) Match by training-token budget instead of FLOPs.
TBD next loop.

### NEXT
- Pick option (a) or (b) for compute matching; document.
- Tokenizer + data pipeline (TinyStories or fineweb-edu sample).
- A first 1000-step training smoke test on baseline (ρ=1).

---

## Loop 3 — Training pipeline + smoke tests

### OBSERVE / ORIENT / DECIDE
Need a working trainer to run anything. Build streaming data loader (TinyStories), full train loop with cosine LR + AdamW + grad clip, optional trackio.

Falsifiable: 60-step CPU run on baseline shows loss decrease ≥0.5 nats; engram path runs without error.

### DEVIL'S ADVOCATE
- Tech: TinyStories chosen for tractable signal at small scale + has both narrative (reasoning) and factual content (memorization). GPT-2 BPE matches our default vocab.
- Tech: Initial bug — was passing `batch[:,:-1]` while model already shifts internally → fixed.
- Exp: Smoke test on CPU not a real result; we need GPU runs.
- Priority: Compute-matching subtlety (P_active varies with ρ) NOT yet resolved. Will be Loop 4 priority. Bias may favor lower-ρ runs which have more total compute due to bigger P_active or vice-versa — must clarify before reporting.

### DO
- `data/loader.py`: streaming token packer.
- `train/train.py`: TrainConfig, cosine LR, AdamW, optional trackio, save state+results.json.
- Smoke test BASELINE (rho=1, d=128 L=4) 60 steps: loss 10.83→9.03 ✓
- Smoke test ENGRAM (rho=0.5, layers [1,2]) 30 steps: loss 10.81→10.02 ✓
- All 17 unit tests still pass.

### RESULT
Trainer works end-to-end on CPU. Engram code-paths exercised. Saved artefacts: `out/smoke_*/results.json`.

### NEXT (decision pending — priority attack from this loop)
- Decide compute-matching: option (a) iso-active by varying depth instead of just FFN width; option (b) match by training-token budget.
- After deciding, plan a small GPU sweep on HF Jobs.
- Eval harness: PPL + at least one knowledge probe.
