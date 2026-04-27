"""
Engram conditional memory module — small-scale faithful reimplementation.

Single-stream variant (hc_mult=1) of the multi-branch design from
DeepSeek's Engram (arXiv 2601.07372). Faithful to the reference demo at
github.com/deepseek-ai/Engram/engram_demo_v1.py, with the following
*intentional* deviations for the sub-1B regime, each justified in
.project/research/small_scale_questions.md:

1. hc_mult=1 by default. We're studying a dense backbone, not a
   hyper-connected multi-branch one. Multi-branch is an orthogonal
   confound for our research question. (See small_scale_questions.md §6.)
2. n_head_per_ngram default 4 (vs 8) — collisions less harmful at
   small table sizes; keeps d_mem manageable. Ablation flag.
3. Hash + memory tables are torch tensors / nn.Parameters end-to-end
   (vs demo's numpy hash → torch embedding) so the whole module is
   GPU-resident and JIT-friendly.
4. Tokenizer compression is plumbed through a `compressed_lookup`
   tensor passed at construction, decoupling Engram from any specific
   tokenizer.

WHAT each component does (paper §2.2-2.4) and WHY:
- Tokenizer compression (collapse case/whitespace/Unicode variants):
  makes the N-gram input space denser so the same hash table covers
  more semantic ground. WHY: the original tokenizer wastes slots on
  surface variants of the same concept ('Apple' vs ' apple').
- Multi-head hashed N-gram retrieval: O(1) lookup of static patterns,
  freeing early transformer layers from reconstructing them.
- Context-aware gate: makes memory *conditional* — when the retrieved
  vector contradicts current context, gate→0 suppresses noise.
- Depthwise causal short-conv with dilation = max_ngram_size: broadens
  the receptive field of retrieved memory without entangling it with
  the immediately-adjacent retrievals (which already share token
  context).

This file is intended to be tested and used. Not a demo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _next_prime(start: int, taken: set) -> int:
    """Smallest prime > start that's not in `taken`."""
    c = start + 1
    while True:
        if c not in taken and _is_prime(c):
            return c
        c += 1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EngramConfig:
    """Engram module config.

    Defaults are tuned for ~100M-300M backbones; override as needed.
    """
    # backbone interface
    hidden_size: int = 512
    # n-gram retrieval
    max_ngram_size: int = 3              # n in {2..max_ngram_size}
    n_head_per_ngram: int = 4
    # per-(n,k) head: roughly this many slots; actual size is the next prime.
    base_table_size: int = 65537
    # per-head embedding dim. d_per_head; total d_mem = (N-1)*K*d_per_head.
    d_per_head: int = 64
    # gating / conv
    kernel_size: int = 4
    use_conv: bool = True
    use_signed_sqrt_gate: bool = True   # demo's stabilizer; ablation flag
    # tokenizer compression
    pad_id: int = 0
    # init scale for embedding table
    table_init_std: float = 0.02
    # placement (used by surrounding model, not by this module)
    layer_ids: List[int] = field(default_factory=lambda: [1, 8])
    # rng for hash multipliers
    seed: int = 0


# ---------------------------------------------------------------------------
# Tokenizer compression: pure id->id map, agnostic to the tokenizer used.
# ---------------------------------------------------------------------------

def build_compression_table(tokenizer) -> torch.LongTensor:
    """Build canonical-id lookup for a HuggingFace tokenizer.

    Spec §A. Decodes each id, applies NFKC + lowercasing + whitespace
    collapse + accent strip, and assigns a new id per unique key.
    Tokens that decode to the unicode replacement char are keyed by
    raw token-piece string (handles byte-level partials).

    Returns: LongTensor of shape (V_orig,) mapping orig id -> canonical id.
    """
    from tokenizers import normalizers, Regex

    sentinel = "\uE000"
    norm = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        # protect single space from Strip
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])

    V = len(tokenizer)
    key2new = {}
    table = [0] * V
    for tid in range(V):
        text = tokenizer.decode([tid], skip_special_tokens=False)
        if "\ufffd" in text:
            key = tokenizer.convert_ids_to_tokens(tid)
        else:
            normed = norm.normalize_str(text)
            key = normed if normed else text
        nid = key2new.get(key)
        if nid is None:
            nid = len(key2new)
            key2new[key] = nid
        table[tid] = nid
    return torch.tensor(table, dtype=torch.long)


# ---------------------------------------------------------------------------
# Engram module
# ---------------------------------------------------------------------------

class EngramMemory(nn.Module):
    """One Engram block, parallel-residual sub-block (single-stream).

    Forward(hidden_states, input_ids) -> [B,L,hidden_size] additive update.

    Caller does:  H = H + EngramMemory(H, input_ids).
    """

    def __init__(
        self,
        cfg: EngramConfig,
        compression_table: torch.LongTensor,
        layer_id: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id

        # ---- Tokenizer compression table (buffer, non-trainable) ----
        # canonical_vocab_size = max(table) + 1
        canon = compression_table.long()
        self.register_buffer("compression_table", canon, persistent=True)
        self.canon_vocab_size = int(canon.max().item()) + 1

        # ---- Hash multipliers (deterministic per-layer odd ints) ----
        # Spec §B. We mirror the demo: per-layer seed = seed + 10007*layer_id.
        rng = torch.Generator().manual_seed(int(cfg.seed) + 10007 * int(layer_id))
        max_long = (1 << 62)  # leave headroom; we'll mod with int64 anyway
        bound = max_long // max(self.canon_vocab_size, 1) // 2
        bound = max(bound, 1)
        # one multiplier per position in the n-gram (max_ngram_size of them)
        mults = torch.randint(low=0, high=int(bound), size=(cfg.max_ngram_size,),
                              dtype=torch.long, generator=rng)
        mults = mults * 2 + 1  # force odd
        self.register_buffer("hash_multipliers", mults, persistent=True)

        # ---- Per-head moduli: consecutive primes >= base_table_size,
        # globally unique across (n, k) for this layer. ----
        head_primes: List[List[int]] = []
        taken: set = set()
        start = cfg.base_table_size - 1
        for n in range(2, cfg.max_ngram_size + 1):
            row = []
            for _ in range(cfg.n_head_per_ngram):
                p = _next_prime(start, taken)
                taken.add(p)
                row.append(p)
                start = p
            head_primes.append(row)
        # Flatten in concat order: n=2 heads 0..K-1, n=3 heads 0..K-1, ...
        flat_primes = [p for row in head_primes for p in row]
        # Offsets into the fused embedding table.
        offsets = [0]
        for p in flat_primes[:-1]:
            offsets.append(offsets[-1] + p)
        total_slots = sum(flat_primes)
        self.register_buffer(
            "head_primes",
            torch.tensor(flat_primes, dtype=torch.long), persistent=True,
        )
        self.register_buffer(
            "head_offsets",
            torch.tensor(offsets, dtype=torch.long), persistent=True,
        )
        self.total_slots = total_slots
        self.num_heads_total = len(flat_primes)  # (N-1) * K

        # ---- Memory table ----
        self.embedding = nn.Embedding(total_slots, cfg.d_per_head)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=cfg.table_init_std)

        # d_mem dimension after concat of all heads
        d_mem = self.num_heads_total * cfg.d_per_head

        # ---- K, V projections + RMSNorms for the gate ----
        d = cfg.hidden_size
        self.W_V = nn.Linear(d_mem, d, bias=False)
        self.W_K = nn.Linear(d_mem, d, bias=False)
        self.q_norm = nn.RMSNorm(d)
        self.k_norm = nn.RMSNorm(d)

        # ---- Short conv ----
        if cfg.use_conv:
            self.conv = nn.Conv1d(
                in_channels=d, out_channels=d,
                kernel_size=cfg.kernel_size,
                groups=d,
                bias=False,
                padding=(cfg.kernel_size - 1) * cfg.max_ngram_size,
                dilation=cfg.max_ngram_size,
            )
            self.conv_norm = nn.RMSNorm(d)
        else:
            self.conv = None

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------
    def _compress(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        return self.compression_table[input_ids]

    def _ngram_hashes(self, canon_ids: torch.LongTensor) -> torch.LongTensor:
        """Compute per-position hash indices for all (n, k) heads.

        Args:
            canon_ids: [B, L] canonical token ids (post-compression).

        Returns:
            [B, L, num_heads_total] int64 indices into the fused embedding
            table (already shifted by per-head offsets).
        """
        B, L = canon_ids.shape
        cfg = self.cfg
        pad_id_compressed = int(self.compression_table[cfg.pad_id].item())

        # left-pad so position t has tokens [t-(n-1)..t]; OOB -> pad
        # base_shifts[k] = canon_ids shifted right by k (pad on left).
        base_shifts = []
        for k in range(cfg.max_ngram_size):
            if k == 0:
                base_shifts.append(canon_ids)
            else:
                pad = canon_ids.new_full((B, k), pad_id_compressed)
                shifted = torch.cat([pad, canon_ids[:, :-k]], dim=1)
                base_shifts.append(shifted)

        all_idx: List[torch.Tensor] = []
        head_cursor = 0
        for n in range(2, cfg.max_ngram_size + 1):
            mix = base_shifts[0] * self.hash_multipliers[0]
            for k in range(1, n):
                mix = torch.bitwise_xor(mix, base_shifts[k] * self.hash_multipliers[k])
            for _ in range(cfg.n_head_per_ngram):
                mod = int(self.head_primes[head_cursor].item())
                offset = int(self.head_offsets[head_cursor].item())
                # Use abs to ensure non-negative; mix can be negative due to overflow.
                idx = (mix.abs() % mod) + offset
                all_idx.append(idx)
                head_cursor += 1
        return torch.stack(all_idx, dim=-1)  # [B, L, num_heads_total]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self,
                hidden_states: torch.Tensor,
                input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, hidden_size] from backbone (pre-block H^(l)).
            input_ids: [B, L] raw token ids (pre-compression).

        Returns:
            [B, L, hidden_size] additive contribution. Caller adds to H.
        """
        B, L, D = hidden_states.shape
        canon = self._compress(input_ids)                 # [B, L]
        idx = self._ngram_hashes(canon)                   # [B, L, H_total]
        embs = self.embedding(idx)                        # [B, L, H_total, d_per_head]
        e = embs.flatten(start_dim=-2)                    # [B, L, d_mem]

        v = self.W_V(e)                                   # [B, L, D]
        k = self.W_K(e)                                   # [B, L, D]

        q_n = self.q_norm(hidden_states)
        k_n = self.k_norm(k)
        # scalar gate: ⟨q,k⟩ / sqrt(D), then signed-sqrt squashing, then sigmoid.
        dot = (q_n * k_n).sum(dim=-1) / math.sqrt(D)      # [B, L]
        if self.cfg.use_signed_sqrt_gate:
            dot = dot.sign() * dot.abs().clamp_min(1e-6).sqrt()
        alpha = torch.sigmoid(dot).unsqueeze(-1)          # [B, L, 1]
        v_tilde = alpha * v                               # [B, L, D]

        if self.conv is not None:
            x = self.conv_norm(v_tilde).transpose(1, 2)   # [B, D, L]
            y = self.conv(x)[:, :, :L].transpose(1, 2)    # causal trim
            y = F.silu(y)
            return v_tilde + y
        return v_tilde
