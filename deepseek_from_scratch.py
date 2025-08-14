# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# deepseek_from_scratch.py
# ----------------------------------
# Educational, runnable Python code that mirrors the architecture taught in the
# "Build DeepSeek from Scratch" : Transformer blocks with
# RMSNorm, Rotary Positional Embeddings (RoPE), Grouped-Query Attention (GQA-like),
# SwiGLU feed-forward, and a Mixture-of-Experts (MoE) layer with top-2 gating.
# Includes a minimal KV-cache for autoregressive inference and a simple training
# scaffold.

# Notes
# -----
# - This is an educational implementation tailored for clarity and readability.
# - It is *not* the official DeepSeek implementation. It aims to capture the key
#   ideas found commonly across modern LLMs (e.g., DeepSeek-V3/R1 motifs: RoPE,
#   MLA, MoE + router, RMSNorm, weight tying, etc.).
# - Replace the included toy tokenizer with your own (e.g., tiktoken/BPE) for real use.
# - PyTorch only; no external frameworks.

# Quickstart
# ----------
# pip install torch
# python deepseek_from_scratch.py --demo

# For tiny training on toy data:
# python deepseek_from_scratch.py --train_tiny

# Author: ProbingBug(Anupam),IIT BOMBAY
# 
# """

from __future__ import annotations
import math
import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



# Utilities


def _gelu(x: torch.Tensor) -> torch.Tensor:
    # F.gelu is fine; this is a stable alias
    return F.gelu(x)


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (no mean-centering), a common LLM norm.
    y = x * w / rms(x), where rms(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


# Rotary Positional Embeddings (RoPE)

class RotaryEmbedding(nn.Module):
    """Applies RoPE to query and key tensors.
    Reference: Su et al. (RoFormer), widely used in modern LLMs.
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim = dim
        self.base = base

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: [B, T, H, Dh] where Dh is head_dim
        seq_positions: [T] integer positions
        """
        device = q.device
        T = seq_positions.shape[0]
        half = self.dim // 2

        # Frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half, device=device).float() / half))  # [half]
        # angles: [T, half]
        angles = torch.einsum('t,d->td', seq_positions.float(), inv_freq)

        # [T, 1, 1, half]
        sin = angles.sin()[..., None, None, :]
        cos = angles.cos()[..., None, None, :]

        def apply_rope(x):
            # x: [B, T, H, Dh]
            x1 = x[..., :half]
            x2 = x[..., half:]
            x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            return x_rot

        return apply_rope(q), apply_rope(k)



# SwiGLU MLP


class SwiGLU(nn.Module):
    """SwiGLU feed-forward as in many LLMs (e.g., PaLM/LLAMA family variants).
    FFN(x) = W3( (W1 x) * swish(W2 x) )
    Hidden dim typically 2-4x the model dim.
    """
    def __init__(self, dim: int, hidden_mult: float = 4.0, bias: bool = False):
        super().__init__()
        hidden = int(dim * hidden_mult)
        # split into two linear projections of size hidden each
        self.w1 = nn.Linear(dim, hidden, bias=bias)
        self.w2 = nn.Linear(dim, hidden, bias=bias)
        self.w3 = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)           # gate
        b = self.w2(x)           # up
        return self.w3(b * F.silu(a))



# Top-2 Router for MoE


class Top2Router(nn.Module):
    """Simple top-2 gating for MoE with a capacity factor.
    Returns routed outputs and auxiliary load-balancing loss.
    """
    def __init__(self, model_dim: int, num_experts: int, capacity_factor: float = 1.25):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.linear = nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, C]
        Returns:
            combined: [B, T, C] mixed expert outputs
            aux_loss: scalar LB loss to encourage balanced routing
        """
        B, T, C = x.shape
        logits = self.linear(x)  # [B, T, E]
        gates = F.softmax(logits, dim=-1)  # [B, T, E]

        # top-2 per token
        topk = torch.topk(gates, k=2, dim=-1)
        indices = topk.indices  # [B, T, 2]
        scores = topk.values    # [B, T, 2]

        # capacity per expert
        E = self.num_experts
        capacity = max(1, int(self.capacity_factor * (B * T * 2) / E))

        # dispatch
        # For simplicity: flatten tokens, then bucket by expert and take first 'capacity'
        x_flat = x.view(B * T, C)
        idx_flat = indices.view(B * T, 2)
        sc_flat = scores.view(B * T, 2)

        # Collect per-expert lists
        expert_inputs = [list() for _ in range(E)]
        expert_scales = [list() for _ in range(E)]
        positions = [list() for _ in range(E)]  # positions to scatter back

        for i in range(B * T):
            for j in range(2):
                e = int(idx_flat[i, j])
                if len(expert_inputs[e]) < capacity:
                    expert_inputs[e].append(x_flat[i])
                    expert_scales[e].append(sc_flat[i, j])
                    positions[e].append(i)

        # Prepare tensors per expert, apply experts, then scatter back
        outputs = torch.zeros_like(x_flat)
        load = gates.mean(dim=(0, 1))  # [E] expected load
        for e in range(E):
            if len(expert_inputs[e]) == 0:
                continue
            xin = torch.stack(expert_inputs[e], dim=0)           # [Ne, C]
            scale = torch.stack(expert_scales[e], dim=0)[:, None]  # [Ne, 1]
            y = self.experts[e](xin)                              # [Ne, C]
            y = y * scale
            pos = torch.tensor(positions[e], device=x.device, dtype=torch.long)
            outputs.index_copy_(0, pos, outputs.index_select(0, pos) + y)

        combined = outputs.view(B, T, C)

        # Auxiliary loss: encourage uniform expert usage
        # (Eq: entropy-style or squared diff from uniform)
        target = torch.full_like(load, 1.0 / E)
        aux_loss = F.mse_loss(load, target)

        return combined, aux_loss

    def set_experts(self, experts: nn.ModuleList):
        self.experts = experts


class ExpertMLP(nn.Module):
    """A per-expert MLP (we reuse SwiGLU core)."""
    def __init__(self, dim: int, hidden_mult: float = 4.0, bias: bool = False):
        super().__init__()
        self.ff = SwiGLU(dim, hidden_mult=hidden_mult, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class MoE(nn.Module):
    """Mixture-of-Experts with Top-2 routing and capacity limit."""
    def __init__(self, dim: int, num_experts: int = 4, hidden_mult: float = 4.0, capacity_factor: float = 1.25):
        super().__init__()
        self.router = Top2Router(dim, num_experts=num_experts, capacity_factor=capacity_factor)
        experts = [ExpertMLP(dim, hidden_mult=hidden_mult) for _ in range(num_experts)]
        self.experts = nn.ModuleList(experts)
        self.router.set_experts(self.experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.router(x)



# Grouped-Query Attention (GQA-like) with RoPE + KV Cache


class KVCache:
    """A simple KV cache for autoregressive decoding."""
    def __init__(self):
        self.key = None   # [B, T, Hkv, Dh]
        self.value = None # [B, T, Hkv, Dh]

    def append(self, k: torch.Tensor, v: torch.Tensor):
        if self.key is None:
            self.key = k
            self.value = v
        else:
            self.key = torch.cat([self.key, k], dim=1)
            self.value = torch.cat([self.value, v], dim=1)


class Attention(nn.Module):
    """Multi-Head Attention(MLA) with grouped K/V heads and RoPE."""
    def __init__(self, dim: int, num_heads: int, kv_heads: Optional[int] = None, rope_base: float = 10000.0, bias: bool = False, attn_dropout: float = 0.0, resid_dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kv_heads = kv_heads if kv_heads is not None else num_heads
        assert self.num_heads % self.kv_heads == 0, "num_heads must be divisible by kv_heads"

        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(dim, dim, bias=bias)

        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, x: torch.Tensor, cache: Optional[KVCache] = None, positions: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        H = self.num_heads
        Hkv = self.kv_heads
        Dh = self.head_dim
        group = H // Hkv

        q = self.q_proj(x).view(B, T, H, Dh)
        k = self.k_proj(x).view(B, T, Hkv, Dh)
        v = self.v_proj(x).view(B, T, Hkv, Dh)

        if positions is None:
            positions = torch.arange(0, T, device=x.device, dtype=torch.long)

        # Apply RoPE to q,k
        # For GQA: RoPE per KV head; we broadcast later to Q groups
        q = q.view(B, T, Hkv, group, Dh).transpose(2, 3)  # [B, T, group, Hkv, Dh]
        q, k = self.rope(q, k, positions)                 # apply on latent Hkv dimension
        q = q.transpose(2, 3).contiguous().view(B, T, H, Dh)  # back to [B, T, H, Dh]

        # KV cache (autoregressive)
        if cache is not None:
            cache.append(k, v)
            k_all = cache.key
            v_all = cache.value
            Tkv = k_all.size(1)
            pos = torch.arange(Tkv - 1, Tkv, device=x.device, dtype=torch.long)
            # mask will be handled below
        else:
            k_all, v_all = k, v
            Tkv = T

        # Expand kv to match q heads via repeat_interleave over groups
        k_all = k_all.repeat_interleave(group, dim=2)  # [B, Tkv, H, Dh]
        v_all = v_all.repeat_interleave(group, dim=2)  # [B, Tkv, H, Dh]

        # scaled dot-product attention
        att = torch.einsum('bthd,bshd->bhts', q, k_all) / math.sqrt(Dh)  # [B, H, T, Tkv]

        # causal mask
        if mask is None:
            mask = torch.ones(T, Tkv, device=x.device, dtype=torch.bool).tril(diagonal=0)
        att = att.masked_fill(~mask[None, None, :, :], float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = torch.einsum('bhts,bshd->bthd', att, v_all)  # [B, T, H, Dh]
        y = y.contiguous().view(B, T, C)
        y = self.o_proj(y)
        y = self.resid_dropout(y)
        return y



# Transformer Block


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, kv_heads: Optional[int], mlp_hidden_mult: float, dropout: float, use_moe: bool = False, num_experts: int = 4, moe_capacity: float = 1.25):
        super().__init__()
        self.use_moe = use_moe
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, kv_heads=kv_heads, attn_dropout=dropout, resid_dropout=dropout)
        self.ff_norm = RMSNorm(dim)
        if use_moe:
            self.ff = MoE(dim, num_experts=num_experts, hidden_mult=mlp_hidden_mult, capacity_factor=moe_capacity)
        else:
            self.ff = SwiGLU(dim, hidden_mult=mlp_hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cache: Optional[KVCache] = None, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention
        a = self.attn_norm(x)
        a = self.attn(a, cache=cache, mask=mask)
        x = x + a

        # FF / MoE
        b = self.ff_norm(x)
        if self.use_moe:
            b, aux_loss = self.ff(b)  # MoE returns (y, aux_loss)
        else:
            b = self.ff(b)
            aux_loss = x.new_zeros(())

        x = x + self.dropout(b)
        return x, aux_loss



# Model Config


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    kv_heads: Optional[int] = 4  # grouped-query attention (e.g., 8 heads with 4 kv heads)
    mlp_hidden_mult: float = 4.0
    dropout: float = 0.0
    max_seq_len: int = 2048
    use_moe_every: int = 0   # set to N to use MoE in every Nth layer (e.g., 2 => layers 2,4,6,...)
    num_experts: int = 4
    moe_capacity: float = 1.25
    tie_weights: bool = True



# Transformer Language Model


class DeepSeekStyleLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_buffer = torch.arange(0, cfg.max_seq_len, dtype=torch.long)
        self.blocks = nn.ModuleList()
        for i in range(cfg.num_layers):
            use_moe = (cfg.use_moe_every > 0) and ((i + 1) % cfg.use_moe_every == 0)
            block = TransformerBlock(
                dim=cfg.dim,
                num_heads=cfg.num_heads,
                kv_heads=cfg.kv_heads,
                mlp_hidden_mult=cfg.mlp_hidden_mult,
                dropout=cfg.dropout,
                use_moe=use_moe,
                num_experts=cfg.num_experts,
                moe_capacity=cfg.moe_capacity,
            )
            self.blocks.append(block)

        self.norm_f = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.embed.weight = self.lm_head.weight  # weight tying

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, use_cache: bool = False):
        """
        idx: [B, T] token IDs
        Returns:
            logits: [B, T, V]
            loss (optional): cross-entropy
            aux_losses: dict with 'moe_aux' if applicable
        """
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, "Sequence length exceeds model max_seq_len"

        x = self.embed(idx)  # [B, T, C]
        mask = torch.ones(T, T, device=idx.device, dtype=torch.bool).tril()

        aux_losses = []
        caches = [KVCache() if use_cache else None for _ in self.blocks]

        for i, block in enumerate(self.blocks):
            x, aux = block(x, cache=caches[i], mask=mask)
            if aux.numel() > 0:
                aux_losses.append(aux)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        aux_total = None
        if len(aux_losses) > 0:
            aux_total = torch.stack(aux_losses).mean()

        return logits, loss, {"moe_aux": aux_total}

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 50, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
        """
        Autoregressive generation using KV cache.
        """
        self.eval()
        device = next(self.parameters()).device
        idx = idx.to(device)
        B, T = idx.shape

        caches = [KVCache() for _ in self.blocks]

        for _ in range(max_new_tokens):
            x = self.embed(idx[:, -1:])  # only the last token at each step
            mask = torch.ones(1, 1 + sum(c.key.size(1) if c.key is not None else 0 for c in caches), device=device, dtype=torch.bool)
            mask = mask.tril()

            for i, block in enumerate(self.blocks):
                x, _ = block(x, cache=caches[i], mask=mask)

            x = self.norm_f(x)
            logits = self.lm_head(x)[:, -1, :] / max(1e-8, temperature)  # [B, V]

            # top-k / nucleus
            if top_k is not None:
                vals, inds = torch.topk(logits, k=top_k, dim=-1)
                filt = torch.full_like(logits, float('-inf'))
                filt.scatter_(1, inds, vals)
                logits = filt
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumprobs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                mask_p = cumprobs > top_p
                mask_p[..., 1:] = mask_p[..., :-1].clone()
                mask_p[..., 0] = False
                sorted_logits[mask_p] = float('-inf')
                logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat([idx, next_token], dim=1)

        return idx



# Toy Tokenizer (byte-level placeholder)


class ByteTokenizer:
    """A minimal byte-level tokenizer mapping bytes to 0..255 and back.
    Replace with a real BPE/WordPiece for practical use.
    """
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> torch.Tensor:
        data = text.encode('utf-8', errors='replace')
        ids = torch.tensor(list(data), dtype=torch.long)
        return ids

    def decode(self, ids: torch.Tensor) -> str:
        data = bytes([int(i) for i in ids])
        return data.decode('utf-8', errors='replace')



# Training Scaffold


def tiny_training_demo(device: str = "cpu"):
    """
    A tiny overfit-on-constant-string demo to verify the pipeline.
    """
    tokenizer = ByteTokenizer()
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        dim=128,
        num_layers=4,
        num_heads=8,
        kv_heads=4,
        mlp_hidden_mult=4.0,
        dropout=0.0,
        max_seq_len=128,
        use_moe_every=2,  # every 2nd layer uses MoE
        num_experts=4,
        moe_capacity=1.25,
    )
    model = DeepSeekStyleLM(cfg).to(device)

    text = "DeepSeek-from-scratch tiny training demo! "
    data = tokenizer.encode(text * 64)  # longer text by repetition
    seq_len = 64
    bs = 8

    def get_batch():
        ix = torch.randint(0, len(data) - seq_len - 1, (bs,))
        x = torch.stack([data[i:i+seq_len] for i in ix])
        y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
        return x.to(device), y.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for step in range(200):
        x, y = get_batch()
        logits, loss, aux = model(x, targets=y)
        total_loss = loss
        if aux["moe_aux"] is not None:
            total_loss = total_loss + 1e-2 * aux["moe_aux"]
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"step {step+1:4d} | loss {loss.item():.4f} | moe_aux {float(aux['moe_aux']) if aux['moe_aux'] is not None else 0.0:.4f}")

    # Generate a short sample
    model.eval()
    prompt = "DeepSeek:"
    idx = tokenizer.encode(prompt)[None, :].to(device)
    out = model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=50)
    print("=== SAMPLE ===")
    print(tokenizer.decode(out[0].cpu()))


def build_model_for_resume(vocab_size: int = 32000) -> DeepSeekStyleLM:
    """
    Factory: returns a moderately small model suitable to list on resume.
    """
    cfg = ModelConfig(
        vocab_size=vocab_size,
        dim=512,
        num_layers=8,
        num_heads=8,
        kv_heads=4,
        mlp_hidden_mult=4.0,
        dropout=0.0,
        max_seq_len=1024,
        use_moe_every=2,      # MoE on layers 2,4,6,8
        num_experts=4,
        moe_capacity=1.25,
        tie_weights=True,
    )
    return DeepSeekStyleLM(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run tiny training demo and generate sample text.")
    parser.add_argument("--train_tiny", action="store_true", help="Alias for --demo.")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.demo or args.train_tiny:
        tiny_training_demo(device=args.device)
    else:
        model = build_model_for_resume()
        n_params = sum(p.numel() for p in model.parameters())
        print(model)
        print(f"\nModel params: {n_params/1e6:.2f}M")


if __name__ == "__main__":
    main()
