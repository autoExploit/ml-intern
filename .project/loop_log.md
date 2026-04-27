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

---

## Loop 4 — Eval harness

### OBSERVE
Loops 0–3 left us with a working module + backbone + trainer; smoke runs
green on CPU but undertrained. Compute-matching design (iso-active vs
iso-param) coded but not documented. No held-out evaluator beyond raw loss.

### ORIENT
Highest-leverage gap: there is *no way to detect the knowledge-vs-reasoning
split* that is Engram's central claim. A ρ-sweep without probes produces
only PPL curves, which collapse the very signal we care about. Building
the eval harness is upstream of every future experiment.

### DECIDE
Build (a) PPL evaluator over a held-out token stream, (b) two synthetic
probes — NameRecall (knowledge / static-pattern memory proxy) and
InductionCopy (reasoning-circuit / induction-head proxy), (c) an end-to-end
runner that loads any checkpoint and writes `eval.json`. Document the
two-sweep compute-matching decision in `.project/research/`.

Falsifiable: (i) `pytest engram_micro/tests` stays green and gains ≥3 tests;
(ii) `run_eval` produces a finite PPL and 0–1 accuracy on the existing
smoke checkpoint; (iii) doc enumerates which scientific question each
sweep answers.

### DEVIL'S ADVOCATE
- *Tech*: Probe tokenization is fragile — leading-space vs not, multi-BPE
  names. Mitigation: probe takes the *first BPE id of the with-leading-space
  form* when prefix doesn't end in space, and matches that single id. Names
  in DEFAULT_NAMES were chosen to be common GPT-2 single-token surface forms.
- *Tech*: InductionCopy uses random rare tokens. The model's argmax is
  almost always a high-frequency token, so top-1 will be ≈0 for any small
  model on this probe — making it look like a useless metric. Mitigation:
  also report mean rank of gold and mean log-prob of gold; rank shifts long
  before top-1 does. Reported in ProbeResult.
- *Exp*: Smoke checkpoints are at chance — running the harness on them
  doesn't validate that the harness can detect the Engram effect, only that
  it runs. Accepted: this is an infrastructure loop, not a science loop.
  Future loops will validate sensitivity on properly-trained models.
- *Priority*: Should we instead build the sweep runner first? No — it would
  produce only loss curves until the harness exists. Eval is upstream.

Rebuttals stand. Proceeded.

### DO
- `engram_micro/eval/perplexity.py` — token-weighted PPL.
- `engram_micro/eval/probes.py` — NameRecall, InductionCopy, ProbeResult,
  bootstrap CI, careful BPE handling.
- `engram_micro/eval/run_eval.py` — checkpoint loader + report writer.
- `engram_micro/tests/test_eval.py` — 4 new tests (PPL runs, both probes
  run, determinism).
- `.project/research/compute_matching.md` — decision: run two sweeps
  (iso-active and iso-param), report both, with explicit reasoning about
  which scientific question each answers.

### RESULT
- 22/22 unit tests pass (was 18; +4 eval tests).
- `run_eval` on `out/smoke_baseline/last.pt` produces:
    PPL ≈ 8507 (60-step CPU smoke run, expected high)
    NameRecall top-1 = 0.00, mean_rank ≈ 19500
    InductionCopy top-1 = 0.00, mean_rank ≈ 30000
  All numbers near chance for an undertrained tiny model — code path
  validated, signal not yet present.
- Compute-matching strategy now formal: Sweep A iso-active + Sweep B iso-param,
  with rationale for what each isolates.

### NEXT
- Sweep runner script (`scripts/sweep_rho.py`) that iterates ρ ∈
  {0.0, 0.25, 0.5, 0.74, 1.0} for each design_mode, dispatches training,
  collects results to `.project/results/sweeps/<mode>/<rho>.json`.
- A real (longer) baseline run on GPU. Local box is CPU-only — plan to
  invoke HF Jobs (existing repo scaffolding) or accept CPU-budget runs.
- Sensitivity check: train two ρ values on TinyStories long enough to
  exit chance regime, verify probes show non-zero spread.
