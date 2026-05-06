"""Unit tests for Engram module — shape, determinism, gradient flow,
tokenizer compression, hash uniqueness/coverage, causality."""
import torch
import pytest

from engram_micro.model.engram import (
    EngramConfig,
    EngramMemory,
    build_compression_table,
    _next_prime,
    _is_prime,
)


def _identity_table(V: int) -> torch.LongTensor:
    return torch.arange(V, dtype=torch.long)


def _make(layer_id=1, V=257, **kw):
    cfg = EngramConfig(
        hidden_size=64,
        max_ngram_size=3,
        n_head_per_ngram=2,
        base_table_size=257,
        d_per_head=16,
        kernel_size=3,
        pad_id=0,
        seed=0,
        **kw,
    )
    return EngramMemory(cfg, _identity_table(V), layer_id=layer_id)


def test_prime_helpers():
    assert _is_prime(2) and _is_prime(257) and not _is_prime(1) and not _is_prime(4)
    assert _next_prime(10, set()) == 11
    assert _next_prime(10, {11}) == 13


def test_forward_shape():
    m = _make()
    B, L = 2, 7
    h = torch.randn(B, L, 64)
    ids = torch.randint(0, 256, (B, L))
    out = m(h, ids)
    assert out.shape == (B, L, 64)


def test_forward_determinism():
    m = _make()
    m.eval()
    ids = torch.randint(0, 256, (1, 5))
    h = torch.randn(1, 5, 64)
    a = m(h, ids)
    b = m(h, ids)
    assert torch.allclose(a, b)


def test_grad_flows_into_table_and_projs():
    # Use nonzero table init so W_V/W_K receive nonzero input at step 0;
    # under the (new) default zero-init, grads to W_V/W_K are 0 at step 0
    # by construction (input is identically zero) — see Loop 6 init change.
    m = _make(table_init_std=0.02)
    h = torch.randn(2, 5, 64, requires_grad=True)
    ids = torch.randint(1, 256, (2, 5))
    out = m(h, ids).sum()
    out.backward()
    assert m.embedding.weight.grad is not None
    assert m.embedding.weight.grad.abs().sum() > 0
    assert m.W_V.weight.grad.abs().sum() > 0
    assert m.W_K.weight.grad.abs().sum() > 0


def test_pad_handling_short_prefix():
    """Position 0 has no left context; should still produce valid output
    (the pad token's compressed id fills missing prefix slots)."""
    m = _make()
    ids = torch.zeros(1, 4, dtype=torch.long)
    ids[0, 0] = 5
    ids[0, 1] = 7
    out = m(torch.randn(1, 4, 64), ids)
    assert torch.isfinite(out).all()


def test_hash_indices_in_range():
    m = _make()
    ids = torch.randint(0, 256, (3, 9))
    canon = m._compress(ids)
    idx = m._ngram_hashes(canon)
    # every column h must lie in [offset_h, offset_h + prime_h)
    for h in range(m.num_heads_total):
        off = int(m.head_offsets[h].item())
        prime = int(m.head_primes[h].item())
        col = idx[..., h]
        assert (col >= off).all() and (col < off + prime).all()
    # within global table bounds
    assert idx.max().item() < m.embedding.num_embeddings


def test_different_layers_use_different_hash_multipliers():
    m1 = _make(layer_id=1)
    m2 = _make(layer_id=8)
    assert not torch.equal(m1.hash_multipliers, m2.hash_multipliers)


def test_signed_sqrt_ablation_runs():
    m = _make(use_signed_sqrt_gate=False)
    out = m(torch.randn(1, 4, 64), torch.randint(0, 256, (1, 4)))
    assert out.shape == (1, 4, 64)


def test_no_conv_ablation_runs():
    m = _make(use_conv=False)
    out = m(torch.randn(1, 4, 64), torch.randint(0, 256, (1, 4)))
    assert out.shape == (1, 4, 64)
    assert m.conv is None


def test_causality_via_input_perturbation():
    """If we change input_ids[t], outputs[<t] must be unchanged.
    Engram retrieval at position t uses tokens [t-(N-1)..t] only,
    and the conv is causal. So outputs at positions < t must be identical."""
    torch.manual_seed(0)
    m = _make()
    m.eval()
    ids = torch.randint(1, 256, (1, 12))
    h = torch.randn(1, 12, 64)
    a = m(h, ids)
    ids2 = ids.clone()
    ids2[0, 8] = (ids2[0, 8] + 13) % 256
    b = m(h, ids2)
    # positions strictly less than 8 unchanged
    assert torch.allclose(a[0, :8], b[0, :8], atol=1e-6), (
        (a[0, :8] - b[0, :8]).abs().max().item()
    )


def test_compression_table_collapses_case_and_whitespace():
    """Build compression table from a real tokenizer; verify some
    canonical collapses we expect from NFKC+lower+strip."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    table = build_compression_table(tok)
    # 'A' and 'a' should collapse if both are single tokens
    aA = tok.encode("A", add_special_tokens=False)
    aa = tok.encode("a", add_special_tokens=False)
    if len(aA) == 1 and len(aa) == 1:
        assert table[aA[0]] == table[aa[0]]
    # ' a' and ' A'
    sA = tok.encode(" A", add_special_tokens=False)
    sa = tok.encode(" a", add_special_tokens=False)
    if len(sA) == 1 and len(sa) == 1:
        assert table[sA[0]] == table[sa[0]]
    # canon vocab strictly smaller than original
    assert int(table.max()) + 1 < len(tok)


def test_hash_distribution_coverage():
    """Random ids should hit a non-trivial fraction of slots — sanity
    check that primes + multipliers don't cluster pathologically."""
    m = _make()
    ids = torch.randint(0, 256, (16, 256))
    canon = m._compress(ids)
    idx = m._ngram_hashes(canon)
    # for head 0, count unique addresses
    unique = idx[..., 0].unique().numel()
    prime0 = int(m.head_primes[0].item())
    # with 16*256=4096 trials and prime ~257, expect near-saturation
    assert unique > 0.5 * min(4096, prime0)


def test_gate_bias_makes_initial_output_small():
    """With default gate_bias_init=-3.0 and table zero-init, the Engram
    contribution at step 0 must be tiny — well below what would
    significantly perturb a residual stream."""
    import torch
    from engram_micro.model.engram import EngramConfig, EngramMemory

    cfg = EngramConfig(hidden_size=64, max_ngram_size=3, n_head_per_ngram=2,
                       base_table_size=257, d_per_head=16, kernel_size=3,
                       pad_id=0, seed=0)
    ctab = torch.arange(257, dtype=torch.long)
    mem = EngramMemory(cfg, ctab, layer_id=0)
    h = torch.randn(2, 16, 64)
    ids = torch.randint(1, 256, (2, 16))
    out = mem(h, ids)
    assert out.shape == h.shape
    # contribution must be small (table is zero-init; gate ≈ sigmoid(-3))
    assert out.abs().max().item() < 0.1, f"engram init out too large: {out.abs().max()}"


def test_gate_bias_is_trainable():
    """gate_bias must accumulate gradient (i.e. be reachable by autograd)."""
    import torch
    from engram_micro.model.engram import EngramConfig, EngramMemory

    cfg = EngramConfig(hidden_size=64, max_ngram_size=3, n_head_per_ngram=2,
                       base_table_size=257, d_per_head=16, kernel_size=3,
                       pad_id=0, seed=0, table_init_std=0.02)  # nonzero so dot != 0
    ctab = torch.arange(257, dtype=torch.long)
    mem = EngramMemory(cfg, ctab, layer_id=0)
    h = torch.randn(1, 8, 64, requires_grad=True)
    ids = torch.randint(1, 256, (1, 8))
    out = mem(h, ids)
    out.sum().backward()
    assert mem.gate_bias.grad is not None
    assert mem.gate_bias.grad.abs().sum().item() > 0
