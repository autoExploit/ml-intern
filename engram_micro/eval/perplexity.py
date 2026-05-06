"""Perplexity evaluation on a held-out token stream.

WHY: PPL is the floor measurement — it tells us whether the model has
learned the data distribution at all. It does *not* separate knowledge
from reasoning, which is why we also have the probes module. PPL is
computed as exp(mean per-token cross-entropy) over a fixed number of
packed sequences, with no padding/masking — straightforward and
reproducible across runs.
"""
from __future__ import annotations

from typing import Iterator, Optional

import math
import torch


@torch.no_grad()
def perplexity(
    model,
    loader_iter: Iterator[torch.Tensor],
    n_iters: int,
    device: str,
) -> dict:
    """Compute mean per-token NLL and PPL.

    Args:
        model: an EngramLM that takes (input_ids, labels=...) and returns
            (logits, loss) where loss is mean cross-entropy over predicted
            tokens (labels[..., 1:] from logits[..., :-1, :]).
        loader_iter: iterator yielding [B, L] LongTensor batches.
        n_iters: number of batches to consume.
        device: target device for inputs.

    Returns dict with keys: nll (float), ppl (float), n_tokens (int).
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for _ in range(n_iters):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        batch = batch.to(device)
        _, loss = model(batch, labels=batch)
        # loss is mean over (B*(L-1)); reconstruct token-weighted sum
        n = batch.size(0) * (batch.size(1) - 1)
        total_nll += float(loss.item()) * n
        total_tokens += n
    if total_tokens == 0:
        return {"nll": float("nan"), "ppl": float("nan"), "n_tokens": 0}
    nll = total_nll / total_tokens
    return {"nll": nll, "ppl": math.exp(nll), "n_tokens": total_tokens}
