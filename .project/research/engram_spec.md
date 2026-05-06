# Engram — Ground-Truth Technical Spec

Sources:
- Paper: arXiv 2601.07372 (DeepSeek). Sections read: 2 (architecture), 3 (sparsity allocation), 4.1 (configs), Appendix A (hyperparameters), Appendix C (compression case study).
- Reference impl: `github.com/deepseek-ai/Engram/engram_demo_v1.py` (single file, 422 lines, demo only — explicitly says it omits standard MHA/MoE/hyperconnection).

This file is ground truth for our reimplementation. Anything we deviate from must be justified and logged.

---

## A. Tokenizer Compression

**Mechanism:** surjective map `P : V → V'`. Built once at init time as a `np.int64[V]` lookup table (`old2new`).

**How it builds the map** (`CompressedTokenizer._build_lookup_table`):
For each token id `tid` in the base tokenizer:
1. Decode the id to text: `tokenizer.decode([tid], skip_special_tokens=False)`.
2. If the decoded text contains the unicode replacement char `�` (i.e. the token is a partial UTF-8 byte and not standalone-decodable), use the *raw token string* (`convert_ids_to_tokens(tid)`) as the canonical key.
3. Otherwise apply a `tokenizers.normalizers.Sequence` pipeline:
   ```
   NFKC → NFD → StripAccents → Lowercase
   → Replace([\t\r\n ]+ → " ")
   → Replace(^ $ → SENTINEL=U+E000)   # protect a single space from Strip
   → Strip
   → Replace(SENTINEL → " ")
   ```
4. Assign new ids in first-seen order.

**Result on DeepSeek-V3 tokenizer (128k):** ~23% reduction in effective vocab (paper Appendix C). E.g. `'a' '\u00e1' '\u00e4' ... ' a' ' A'` all collapse to single canonical id `'a'`.

**Pad handling:** the original pad id is also remapped through the table (`self.pad_id = lookup_table[pad_id]`).

**Where it runs:** inputs to the *Engram lookup only* are compressed. The backbone embedding/LM head still use the original tokenizer ids. So the compression is N-gram-side only.

---

## B. N-gram Hashing

**N values:** `n ∈ {2, ..., max_ngram_size}` with `max_ngram_size=3` in the demo config (i.e. bigrams + trigrams). Confirmed in paper.

**Heads per ngram order:** `n_head_per_ngram=8`. Each (n, k) pair has its own table partition with its own modulus.

**Hash function** (`_get_ngram_hashes`): a multiplicative-XOR hash over the *compressed* ids:

```
mix = x_t * m_0
for k in 1..n-1:
    mix = mix XOR (x_{t-k} * m_k)
hash_{n,k} = mix mod M_{n,k}
```

where `m_0..m_{N-1}` are odd random int64 multipliers seeded by `seed + 10007*layer_id` (so each layer has *different* hashes — different memory tables don't collide on the same N-gram). All heads of a given (layer, n) share the same `mix`; only the modulus `M_{n,k}` differs. So heads differ purely by mod-prime.

**Modulus selection:** `M_{n,k}` are consecutive primes ≥ `engram_vocab_size[n-2]`. They are chosen *globally unique across (layer, n, k)* (`seen_primes`) so that no two head tables alias to the exact same address space size. Default `engram_vocab_size = [129280*5, 129280*5]` = 646400 slots per (n,k) pair (so for the demo a single 2-gram head has 646400 slots; total slots per layer = `2 ngrams × 8 heads × ~646400 ≈ 10.3M`).

**Padding for short prefixes:** when the position has fewer than n tokens of context, missing positions are filled with the (compressed) pad_id, so all positions still produce a valid hash (the early positions just hash a "pad-prefix" pattern).

**Note:** the demo computes hashes in NumPy on CPU then `torch.from_numpy`s them. Production needs this on-GPU.

---

## C. Memory Table

**Per-layer-per-(n,k) table:** `nn.Embedding(M_{n,k}, d_per_head)` where `d_per_head = n_embed_per_ngram // n_head_per_ngram = 512/8 = 64`.

**Layout in demo:** `MultiHeadEmbedding` fuses *all* (n, k) tables for a layer into one big `nn.Embedding(sum(M_{n,k}), d_per_head)` with offset shifts — purely an indexing convenience, no semantic effect.

**Per-table param count:** `M_{n,k} * d_per_head`. For demo defaults: `≈ 646400 * 64 * 16 (heads across n=2,3) = 661M params per Engram layer`. With 2 Engram layers (1 and 15), total Engram params ≈ 1.3B (in this demo config). The paper's 27B model uses different sizes that net to 5.7B Engram params total.

**dtype/init:** standard `nn.Embedding` init (Normal). No explicit special init in the demo. The table is fully gradient-trained — there is no non-gradient update rule. (No EMA, no count-based statistics, no sparse smoothing.)

**Concatenation of retrieved vectors:** for a position, the retrieved memory vector is

```
e_t = concat over n in {2,3}, k in 1..K of e_{t,n,k}  ∈ ℝ^d_mem
d_mem = (max_ngram_size - 1) * n_embed_per_ngram
      = 2 * 512 = 1024  in demo
```

(Note: the paper's eqn shows `||_{n=2}^N ||_{k=1}^K`, but the demo concatenates with `n_embed_per_ngram` already representing the K-head sum, so per-n bundle is 512 = 8*64. Thus `d_mem = (N-1) * 512`. Reading `MultiHeadEmbedding.forward`: returns `[B,L,num_heads_total, d_per_head]`, which is then `.flatten(start_dim=-2)` to `[B,L, num_heads_total * d_per_head] = [B,L,(N-1)*K*d_per_head] = [B,L,(N-1)*n_embed_per_ngram]`.)

---

## D. Context-Aware Gating (and Multi-branch Adaptation)

**Single-stream form (paper eqn 3-5):**
```
k_t = W_K e_t                                   # [d]
v_t = W_V e_t                                   # [d]
α_t = σ( ⟨RMSNorm(h_t), RMSNorm(k_t)⟩ / sqrt(d) )   # scalar
v_tilde_t = α_t * v_t                           # [T,d]
Y = SiLU(Conv1D_dw(RMSNorm(v_tilde))) + v_tilde    # kernel=4, dilation=N (=3)
H ← H + Y
```
Conv1D is depthwise causal, kernel size 4, dilation = `max_ngram_size`.

**Multi-branch form** used in demo (paper eqn 6, hyper-connections, M=`hc_mult=4` branches):
- One shared `W_V` (`value_proj`).
- M distinct `W_K^(m)` (`key_projs[0..M-1]`).
- For each branch m: `α_t^(m) = σ( ⟨RMSNorm_m(h_t^(m)), RMSNorm_m(W_K^(m) e_t)⟩ / sqrt(d) )`.
- `u_t^(m) = α_t^(m) · (W_V e_t)`.
- The depthwise conv operates on the stacked branch output `[B,L,M,d]` (with channels = M*d, groups=M*d → strictly per-channel depthwise).

**One subtle deviation in demo code from the paper:** before sigmoid, the demo applies a **signed sqrt** to the scaled dot product:
```python
gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
gate = gate.sigmoid()
```
This compresses the magnitude of the dot product before sigmoid (effective range `±sqrt(d)` -> `±d^{1/4}`), keeping the gate in a less saturated regime. This is **not in the paper's equations**. We will reproduce it (load-bearing for stable gating) and add it to our deviations log.

**Final output form (multi-branch):** `output = value_branched + ShortConv(value_branched)` (so the residual-into-conv differs from paper eqn 5 which says `Y = SiLU(Conv(RMSNorm(V~))) + V~` — the demo actually does the analogue inside `ShortConv` already). After return, the caller in `TransformerBlock.forward` does `hidden_states = engram(...) + hidden_states`.

---

## E. Layer Integration

**Where:** Engram is a sub-block in `TransformerBlock`, added *first* (before attn, before MoE):
```
H ← H + Engram(H, input_ids)        # only for layers in layer_ids
H ← H + Attn(H)
H ← H + MoE(H)
```
**Layer ids for 27B (paper):** Engram inserted at layers 2 and 15 (1-indexed) of 30. **Demo config:** `layer_ids=[1,15]` — note **0-indexed** in code, so layers 2 and 16 — close enough to paper. (We will use 0-indexed `[1, depth//2]` for our small models.)

**Pre-norm vs post-norm:** demo uses RMSNorm internally on q/k for the gate; the conv has its own RMSNorm. There is no outer pre-norm wrapping the Engram block — the caller adds it raw to the residual.

---

## F. Sparsity Allocation (ρ)

**Definition (paper §3.1):**
- `P_tot` = total trainable params (excluding embed & LM head).
- `P_act` = activated params per token (training FLOPs proxy).
- `P_sparse = P_tot - P_act`.
- `ρ ∈ [0,1]`: `P_MoE_sparse = ρ * P_sparse`, `P_Engram = (1-ρ) * P_sparse`.

**ρ is a parameter ratio, not a FLOP ratio.** Iso-FLOPs is enforced by holding `P_act` fixed. Iso-param is enforced by holding `P_tot` fixed.

**Reported optima:**
- Compute regime C=2e20: optimum near `ρ ≈ 0.75-0.80`.
- Compute regime C=6e20: optimum near `ρ ≈ 0.80`.
- 27B large run: `ρ = 55/(55+...) ≈ 0.74` (72→55 experts, freed params → 5.7B Engram).
- Pure MoE (`ρ=1`) is suboptimal.
- Even at `ρ=0.4` Engram matches pure MoE.

**Small-model adaptation:** since our backbone is *dense* (no MoE), `ρ` cannot be defined the same way. Two viable redefinitions (TBD in next loop):
1. `P_sparse` = budget that we'd otherwise spend on additional FFN width / depth. `ρ=1` → all goes to FFN; `ρ=0` → all goes to Engram.
2. `ρ` = fraction in FFN of dense backbone params vs Engram table params (excluding attn & embed).
We must commit to a concrete definition and implement it consistently.

---

## G. Training / Loss

**Auxiliary losses for Engram:** none in the demo, none mentioned in the paper for the Engram module. (MoE has its standard balance loss; not Engram.)

**LR / WD for table:** not specified in demo. Paper Appendix A doesn't separate. We'll start with a single optimizer group and *measure*; if Engram embeddings under-train, switch to a higher LR & WD=0 group (standard practice for embeddings).

**Gate warm-up:** none mentioned. Worth ablating but not default.

---

## H. Inference / Efficiency

**Training:** tables are sharded across GPUs; All-to-All communication retrieves rows.
**Inference:** tables offloaded to host RAM. Because indices `z_{t,n,k}` depend only on input ids (deterministic, computable as soon as the token is produced), the host *prefetches* row data for upcoming layers in parallel with on-device compute of preceding transformer blocks. Reported overhead **<3%** at scale.

For our edge-device analysis: this prefetch story is the main efficiency claim and must be re-examined when host RAM is the actual bottleneck.

---

## I. Things in the Codebase Not in the Paper

1. **Signed-sqrt before sigmoid in gate** (D above) — not in equations.
2. **Pad token also gets compressed** through the lookup table, so the "padding" hash doesn't collide with a real token's compressed id by accident.
3. **`�`-handling fallback** in tokenizer compression: byte-level partial tokens use raw token-piece string as canonical, *not* the decoded text. This is a real-world hack the paper doesn't mention.
4. **Per-head primes are required globally unique across all (layer,n,k)** — implementation choice, prevents accidental aliasing.
5. **The depthwise conv operates on the branch-multiplied tensor** (channels = `M * d`), not just `d` channels. So the conv has `M * d` independent depthwise filters, one per (branch, feature) pair.
6. **Conv `dilation = max_ngram_size`** (not 1): the conv looks at every-N-th past gated value, plausibly so the conv + N-gram lookup span mostly-disjoint context windows.
7. **Layer ids in the demo are 0-indexed `[1, 15]`** despite the paper text saying "layers 2 and 15".

---

## Quick parameter accounting (for sanity-checking our small-scale design)

Let `K = n_head_per_ngram`, `Nmax = max_ngram_size`, `D = n_embed_per_ngram`, `d_per_head = D/K`, `d = backbone hidden`.

**Per Engram layer:**
- Tables: `(Nmax-1) * K * M_avg * d_per_head` ≈ `2 * 8 * M * 64` = `1024 * M` params  (tiny per-slot).
- Value proj `W_V`: `(Nmax-1)*D × d` = `2*512 × d` = `1024d`.
- Key projs `W_K^(m)`, M of them: `M * 1024 * d`.
- RMSNorms: negligible.
- Conv: `kernel * M * d` (depthwise) — negligible.

So Engram's *compute* params (non-table) per layer ≈ `(1+M) * 1024 * d` ≈ `5 * 1024d = 5120d` for M=4.

For a 300M backbone with d=768, M=4: ~3.9M compute params per Engram layer, two layers ⇒ ~8M. Memory tables dominate.

If we want a 100M backbone + ~50M Engram table:
table params = `1024 * total_M_avg`. For 50M params → ~50k slots per (n,k) head ⇒ very feasible.

---

## Decisions for Engram-Micro implementation

- Single-branch (`hc_mult=1`) by default. Multi-branch hyper-connections are an orthogonal axis we shouldn't conflate with "does Engram help small dense?" Keep optional flag for later ablation.
- `max_ngram_size = 3` (so n ∈ {2,3}).
- `n_head_per_ngram = 4` (smaller than 27B's 8) — collisions are less harmful at small table sizes and it keeps `d_mem` manageable. To be re-ablated.
- Compression: use whatever tokenizer we adopt for the backbone, build the same NFKC+lower+strip-accents+whitespace-collapse map.
- Layer placement: layers `[1, depth//2]` (0-indexed), e.g. `[1,8]` for depth=16. Confirm with placement ablation later.
- Reproduce the **signed-sqrt** before sigmoid (load-bearing, in the reference code).
- All Engram weights in same optimizer group as backbone initially; reassess.
