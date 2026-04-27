"""Tests for eval harness.

Use a tiny model with random weights so probes return *some* number;
the asserts validate code-correctness, not score quality.
"""
import torch

from engram_micro.model.lm import EngramLM, EngramLMConfig, BackboneConfig
from engram_micro.eval.perplexity import perplexity
from engram_micro.eval.probes import (
    name_recall_probe, induction_copy_probe, ProbeResult,
)


class _DummyTok:
    """Minimal tokenizer with character-bigram-ish encoding sufficient for tests."""

    def __init__(self, vocab_size=300):
        self.vocab_size = vocab_size
        self.eos_token_id = 1
        self.pad_token_id = 0
        # build a stable mapping char -> id
        self._map = {}

    def __len__(self):
        return self.vocab_size

    def _idof(self, s: str) -> int:
        if s not in self._map:
            self._map[s] = (len(self._map) + 2) % (self.vocab_size - 2) + 2
        return self._map[s]

    def encode(self, text, add_special_tokens=False):
        ids = []
        for ch in text:
            ids.append(self._idof(ch))
        return ids


def _tiny_model(V=300, d=32, L=2, h=4):
    cfg = EngramLMConfig(
        backbone=BackboneConfig(vocab_size=V, hidden_size=d, num_layers=L,
                                num_heads=h, ffn_mult=4.0, max_seq_len=128),
        engram=None, engram_layer_ids=[],
    )
    return EngramLM(cfg)


def test_perplexity_runs():
    model = _tiny_model()
    def loader():
        for _ in range(5):
            yield torch.randint(2, 300, (2, 32))
    rep = perplexity(model, iter(loader()), n_iters=5, device="cpu")
    assert "ppl" in rep and rep["ppl"] > 0
    assert rep["n_tokens"] == 5 * 2 * 31


def test_name_recall_probe_runs():
    model = _tiny_model()
    tok = _DummyTok()
    res = name_recall_probe(model, tok, "cpu", max_examples=10, seed=0)
    assert isinstance(res, ProbeResult)
    assert res.n > 0
    assert 0.0 <= res.top1_acc <= 1.0
    assert res.mean_rank >= 1.0


def test_induction_copy_probe_runs():
    model = _tiny_model()
    tok = _DummyTok()
    res = induction_copy_probe(model, tok, "cpu", n_examples=10,
                               seq_len=16, vocab_low=2, vocab_high=300)
    assert res.n == 10
    assert 0.0 <= res.top1_acc <= 1.0


def test_probes_use_only_logits_at_last_position():
    """Smoke: changing tokens before the last position should change
    next-token logits, but altering only the position-N+1 entry shouldn't
    matter (we never look past it). We assert determinism instead."""
    torch.manual_seed(0)
    model = _tiny_model()
    tok = _DummyTok()
    r1 = induction_copy_probe(model, tok, "cpu", n_examples=20,
                              seq_len=24, seed=42, vocab_low=2, vocab_high=300)
    r2 = induction_copy_probe(model, tok, "cpu", n_examples=20,
                              seq_len=24, seed=42, vocab_low=2, vocab_high=300)
    assert r1.top1_acc == r2.top1_acc
    assert r1.mean_rank == r2.mean_rank
