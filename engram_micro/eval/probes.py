"""Lightweight evaluation probes for Engram-Micro.

These probes operationalise the Engram paper's central claim that a memory
module accelerates *static-pattern* lookup, freeing the backbone for
*reasoning*. At sub-1B scale we cannot run MMLU/BBH meaningfully, so we
construct synthetic probes whose scores have well-defined mechanistic
interpretations:

1. NameRecall  — knowledge / static-pattern memory proxy.
   Story-style template introduces a named entity once, then re-references
   it. Accuracy = does the model assign max-prob to the introduced name
   when it should be re-mentioned? This is exactly the kind of static
   lookup Engram is supposed to help with.

2. InductionCopy — induction-head / in-context-copy proxy.
   Random rare-token sequence ending in a marker; measure whether the
   model copies the token that previously followed the marker. This tests
   the induction circuit which the paper argues is freed for reasoning
   *because* static patterns are no longer competing for early layers.

Both probes report top-1 accuracy and rank of the gold token, with bootstrap
confidence intervals. They are fast (CPU-friendly) and rerunnable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import math
import random

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_no_special(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _last_token_ids(tokenizer, word: str) -> List[List[int]]:
    """Return all plausible BPE encodings of `word` as it might appear at
    a position. Tries: leading-space, no-leading-space, capitalised."""
    out = []
    for variant in (" " + word, word):
        ids = _encode_no_special(tokenizer, variant)
        if ids:
            out.append(ids)
    return out


@torch.no_grad()
def _next_token_logits(model, input_ids: torch.LongTensor) -> torch.Tensor:
    """Return logits at the FINAL position, [B, V]."""
    logits, _ = model(input_ids, labels=None)
    return logits[:, -1, :]


# ---------------------------------------------------------------------------
# NameRecall probe
# ---------------------------------------------------------------------------

DEFAULT_NAMES = [
    "Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Liam", "Mia", "Noah", "Olive", "Pete",
    "Quinn", "Rose", "Sam", "Tom", "Uma", "Vince", "Wendy", "Xena",
    "Yara", "Zach",
]

NAME_RECALL_TEMPLATES = [
    "Once upon a time, {NAME} went to the park. Later that day, {NAME}",
    "There was a kid named {NAME}. Every morning, {NAME}",
    "{NAME} loved playing with toys. After lunch, {NAME}",
    "A child called {NAME} found a coin. Right away, {NAME}",
    "One sunny day, {NAME} saw a cat. Without thinking, {NAME}",
]


@dataclass
class ProbeResult:
    name: str
    n: int
    top1_acc: float
    mean_rank: float
    mean_logprob_gold: float
    ci95: Tuple[float, float]


def _bootstrap_ci(correct: Sequence[int], n_boot: int = 200, seed: int = 0) -> Tuple[float, float]:
    if not correct:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(correct)
    accs = []
    for _ in range(n_boot):
        s = sum(correct[rng.randrange(n)] for _ in range(n)) / n
        accs.append(s)
    accs.sort()
    return (accs[int(0.025 * n_boot)], accs[int(0.975 * n_boot)])


@torch.no_grad()
def name_recall_probe(
    model,
    tokenizer,
    device: str,
    names: Optional[Sequence[str]] = None,
    templates: Optional[Sequence[str]] = None,
    max_examples: int = 200,
    seed: int = 0,
) -> ProbeResult:
    """Test whether the model recalls a freshly-introduced name.

    For each (name, template), compute logits at the position right after
    the second occurrence of {NAME} in the template — the position where
    the model is *about to emit a continuation*. Convention: we instead
    measure the SECOND {NAME}'s first BPE token. The setup:

        prompt    = template up to and including the position before the
                    name's first BPE token at its second occurrence.
        gold      = the first BPE id of the second occurrence of {NAME}.

    Top-1 = argmax over vocabulary equals gold.

    The probe is *static-pattern* in nature: the same name was mentioned
    one sentence ago. Accuracy at this probe should be helped by Engram
    if the paper's mechanism holds at small scale.
    """
    model.eval()
    names = list(names or DEFAULT_NAMES)
    templates = list(templates or NAME_RECALL_TEMPLATES)

    rng = random.Random(seed)
    pairs = [(t, n) for t in templates for n in names]
    rng.shuffle(pairs)
    pairs = pairs[:max_examples]

    correct = []
    ranks = []
    logprobs = []

    for tpl, name in pairs:
        # find position of SECOND {NAME}
        first_marker = tpl.find("{NAME}")
        second_marker = tpl.find("{NAME}", first_marker + 1)
        if second_marker < 0:
            continue
        prefix = tpl[:second_marker].replace("{NAME}", name)
        # gold: first BPE id of the (with-leading-space) name
        # we don't strip — prefix ends with a space already typically
        gold_variants = _last_token_ids(tokenizer, name)
        if not gold_variants:
            continue
        # pick the variant whose first id matches what comes after `prefix`
        # we choose the leading-space form when prefix ends with non-space.
        if prefix and not prefix.endswith(" "):
            target_text = " " + name
        else:
            target_text = name
        gold_ids = _encode_no_special(tokenizer, target_text)
        if not gold_ids:
            continue
        gold_id = gold_ids[0]

        ids = _encode_no_special(tokenizer, prefix)
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = _next_token_logits(model, x)[0]  # [V]
        argmax = int(logits.argmax().item())
        correct.append(int(argmax == gold_id))
        # rank of gold
        rank = int((logits > logits[gold_id]).sum().item()) + 1
        ranks.append(rank)
        # log-prob of gold (use logsoftmax for stability)
        logprobs.append(float(torch.log_softmax(logits, dim=-1)[gold_id].item()))

    n = len(correct)
    top1 = sum(correct) / n if n else float("nan")
    mean_rank = sum(ranks) / n if n else float("nan")
    mean_lp = sum(logprobs) / n if n else float("nan")
    ci = _bootstrap_ci(correct, seed=seed)
    return ProbeResult(name="NameRecall", n=n, top1_acc=top1,
                       mean_rank=mean_rank, mean_logprob_gold=mean_lp, ci95=ci)


# ---------------------------------------------------------------------------
# InductionCopy probe
# ---------------------------------------------------------------------------

@torch.no_grad()
def induction_copy_probe(
    model,
    tokenizer,
    device: str,
    n_examples: int = 200,
    seq_len: int = 64,
    seed: int = 0,
    vocab_low: int = 1000,
    vocab_high: Optional[int] = None,
) -> ProbeResult:
    """Synthetic in-context copy:

      prefix tokens:  <a, b, c, d, ..., x, y, x>     (random)
      target:         <y>

    where the bigram (x, y) appears once earlier in the sequence and is
    re-cued at the end by repeating x. Tests whether the model can copy
    the previously-seen successor of x. This is the canonical induction-head
    behavior. Engram's gating (which suppresses memory when the local
    bigram is unusual) should not interfere here; the backbone's
    induction circuit does the work.

    Why we still run this probe even though Engram doesn't directly help:
    if Engram is *hurting* induction (e.g., by stealing capacity), this
    probe will catch it. It's a regression detector for the reasoning
    side of the knowledge/reasoning split.
    """
    model.eval()
    rng = random.Random(seed)
    V = vocab_high if vocab_high is not None else min(len(tokenizer), 30000)
    correct = []
    ranks = []
    logprobs = []

    for _ in range(n_examples):
        # build random token sequence of length seq_len-2
        toks = [rng.randrange(vocab_low, V) for _ in range(seq_len - 2)]
        # pick a random index i in [0, seq_len-3) such that the
        # successor toks[i+1] is the gold label
        i = rng.randrange(0, len(toks) - 1)
        x_tok = toks[i]
        y_tok = toks[i + 1]
        # append cue: x_tok again
        seq = toks + [x_tok]
        gold_id = y_tok

        ids = torch.tensor([seq], dtype=torch.long, device=device)
        logits = _next_token_logits(model, ids)[0]
        argmax = int(logits.argmax().item())
        correct.append(int(argmax == gold_id))
        rank = int((logits > logits[gold_id]).sum().item()) + 1
        ranks.append(rank)
        logprobs.append(float(torch.log_softmax(logits, dim=-1)[gold_id].item()))

    n = len(correct)
    top1 = sum(correct) / n if n else float("nan")
    mean_rank = sum(ranks) / n if n else float("nan")
    mean_lp = sum(logprobs) / n if n else float("nan")
    ci = _bootstrap_ci(correct, seed=seed)
    return ProbeResult(name="InductionCopy", n=n, top1_acc=top1,
                       mean_rank=mean_rank, mean_logprob_gold=mean_lp, ci95=ci)
