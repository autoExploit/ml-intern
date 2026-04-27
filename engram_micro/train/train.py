"""Training loop for Engram-Micro.

Minimal but real: AdamW with cosine LR + warmup, gradient clipping,
periodic train/val loss logging, checkpoint save. Optional trackio.

This is the script invoked by HF Jobs and locally for smoke tests.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from engram_micro.model.lm import EngramLM, EngramLMConfig, BackboneConfig
from engram_micro.model.engram import EngramConfig, build_compression_table
from engram_micro.model.accounting import count_params, design_iso_param_configs
from engram_micro.data.loader import make_loader


@dataclass
class TrainConfig:
    # data
    dataset_name: str = "roneneldan/TinyStories"
    train_split: str = "train"
    val_split: str = "validation"
    text_field: str = "text"
    tokenizer_name: str = "gpt2"
    # arch
    rho: float = 1.0                    # 1.0 = no engram baseline
    backbone_d: int = 256
    backbone_layers: int = 8
    backbone_heads: int = 8
    backbone_ffn_mult: float = 4.0
    seq_len: int = 256
    engram_layer_ids: list = field(default_factory=lambda: [1, 4])
    max_engram_table_params: int = 8_000_000
    engram_n_head_per_ngram: int = 4
    engram_d_per_head: int = 64
    engram_max_ngram: int = 3
    # train
    batch_size: int = 16
    max_steps: int = 1000
    warmup_steps: int = 50
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # eval / log
    log_every: int = 10
    eval_every: int = 200
    eval_iters: int = 20
    # io
    out_dir: str = "out/run"
    seed: int = 0
    device: str = "cpu"  # auto-overridden if cuda available
    use_trackio: bool = False
    trackio_project: str = "engram-micro"
    run_name: str = "default"


def _cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def _evaluate(model: EngramLM, loader_iter, n_iters: int, device) -> float:
    model.eval()
    losses = []
    for _ in range(n_iters):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        batch = batch.to(device)
        _, loss = model(batch, labels=batch)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def _build_model(cfg: TrainConfig, tokenizer):
    bb = BackboneConfig(
        vocab_size=len(tokenizer),
        hidden_size=cfg.backbone_d,
        num_layers=cfg.backbone_layers,
        num_heads=cfg.backbone_heads,
        ffn_mult=cfg.backbone_ffn_mult,
        max_seq_len=cfg.seq_len,
    )
    eng = EngramConfig(
        hidden_size=cfg.backbone_d,
        max_ngram_size=cfg.engram_max_ngram,
        n_head_per_ngram=cfg.engram_n_head_per_ngram,
        d_per_head=cfg.engram_d_per_head,
        kernel_size=4,
        pad_id=tokenizer.eos_token_id or 0,
        seed=cfg.seed,
    )
    lm_cfg, info = design_iso_param_configs(
        bb, eng, rho=cfg.rho,
        max_engram_table_params=cfg.max_engram_table_params,
        engram_layer_ids=cfg.engram_layer_ids,
    )
    if lm_cfg.engram is not None:
        ctab = build_compression_table(tokenizer)
    else:
        ctab = None
    model = EngramLM(lm_cfg, compression_table=ctab)
    return model, lm_cfg, info


def train(cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    if cfg.device == "auto" or (cfg.device == "cuda" and torch.cuda.is_available()):
        device = "cuda"
    elif cfg.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, lm_cfg, design_info = _build_model(cfg, tokenizer)
    model = model.to(device)

    counts = count_params(model)
    print(f"[init] device={device}  rho={cfg.rho}  counts={counts}", flush=True)

    train_loader = make_loader(
        cfg.dataset_name, cfg.train_split, tokenizer,
        seq_len=cfg.seq_len, batch_size=cfg.batch_size,
        text_field=cfg.text_field, seed=cfg.seed,
    )

    def fresh_val_iter():
        return make_loader(
            cfg.dataset_name, cfg.val_split, tokenizer,
            seq_len=cfg.seq_len, batch_size=cfg.batch_size,
            text_field=cfg.text_field, seed=cfg.seed + 1,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )

    tio = None
    if cfg.use_trackio:
        try:
            import trackio
            tio = trackio.init(project=cfg.trackio_project, name=cfg.run_name,
                               config={**asdict(cfg), "counts": counts})
        except Exception as e:
            print(f"[trackio] disabled: {e}", flush=True)

    history = []
    t0 = time.time()
    model.train()
    step = 0
    for batch in train_loader:
        if step >= cfg.max_steps:
            break
        lr = _cosine_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr
        batch = batch.to(device)
        logits, loss = model(batch, labels=batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if step % cfg.log_every == 0:
            dt = time.time() - t0
            print(f"step={step:5d}  loss={loss.item():.4f}  lr={lr:.2e}  t={dt:.1f}s",
                  flush=True)
            history.append({"step": step, "loss": loss.item(), "lr": lr})
            if tio:
                trackio.log({"train/loss": loss.item(), "train/lr": lr}, step=step)
        if cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
            vloss = _evaluate(model, fresh_val_iter(), cfg.eval_iters, device)
            print(f"step={step:5d}  val_loss={vloss:.4f}", flush=True)
            history.append({"step": step, "val_loss": vloss})
            if tio:
                trackio.log({"val/loss": vloss}, step=step)
        step += 1

    # final eval
    vloss = _evaluate(model, fresh_val_iter(), cfg.eval_iters, device)
    print(f"[done] step={step}  final_val_loss={vloss:.4f}  t={time.time()-t0:.1f}s",
          flush=True)
    history.append({"step": step, "val_loss_final": vloss})

    # save
    out_path = os.path.join(cfg.out_dir, "last.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "lm_config": {
                "backbone": asdict(lm_cfg.backbone),
                "engram_layer_ids": lm_cfg.engram_layer_ids,
                "engram": asdict(lm_cfg.engram) if lm_cfg.engram else None,
            },
            "design_info": design_info,
            "param_counts": counts,
            "history": history,
        },
        out_path,
    )
    with open(os.path.join(cfg.out_dir, "results.json"), "w") as f:
        json.dump({"counts": counts, "history": history,
                   "final_val_loss": vloss, "rho": cfg.rho}, f, indent=2)
    print(f"[save] -> {out_path}", flush=True)
    if tio:
        try:
            trackio.finish()
        except Exception:
            pass
    return vloss


def parse_args():
    p = argparse.ArgumentParser()
    for k, v in asdict(TrainConfig()).items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=lambda s: s.lower() == "true", default=v)
        elif isinstance(v, list):
            p.add_argument(f"--{k}", type=lambda s: [int(x) for x in s.split(",")],
                           default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(**vars(args))
    train(cfg)
