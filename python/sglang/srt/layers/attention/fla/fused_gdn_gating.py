from typing import Tuple

import torch
import triton
import triton.language as tl


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
# beta_output = b.sigmoid()
@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    stride_a,
    stride_b,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + i_b * stride_a + head_off, mask=mask)
    blk_b = tl.load(b + i_b * stride_b + head_off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta_output.to(b.dtype.element_ty), mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, num_heads = a.shape
    seq_len = 1
    stride_a = a.stride(0)
    stride_b = b.stride(0)
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=torch.float32, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        stride_a,
        stride_b,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output


@triton.jit
def fused_gdn_gating_chunk_cumsum_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    stride_a_t: tl.constexpr,
    stride_a_h: tl.constexpr,
    stride_b_t: tl.constexpr,
    stride_b_h: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    chunk_size: tl.constexpr,
    block_heads: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)
    offs_t = i_t * chunk_size + tl.arange(0, chunk_size)
    offs_h = i_h * block_heads + tl.arange(0, block_heads)
    mask = (offs_t[:, None] < seq_len) & (offs_h[None, :] < num_heads)

    blk_a = tl.load(
        a + offs_t[:, None] * stride_a_t + offs_h[None, :] * stride_a_h,
        mask=mask,
        other=0.0,
    )
    blk_b = tl.load(
        b + offs_t[:, None] * stride_b_t + offs_h[None, :] * stride_b_h,
        mask=mask,
        other=0.0,
    )
    blk_A_log = tl.load(A_log + offs_h, mask=offs_h < num_heads, other=0.0)
    blk_bias = tl.load(dt_bias + offs_h, mask=offs_h < num_heads, other=0.0)

    x = blk_a.to(tl.float32) + blk_bias[None, :].to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log[None, :].to(tl.float32)) * softplus_x
    blk_g = tl.cumsum(blk_g, axis=0)

    out_off = offs_t[:, None] * num_heads + offs_h[None, :]
    tl.store(g + out_off, blk_g, mask=mask)
    tl.store(
        beta_output + out_off,
        tl.sigmoid(blk_b.to(tl.float32)).to(b.dtype.element_ty),
        mask=mask,
    )


def fused_gdn_gating_chunk_cumsum(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    chunk_size: int = 64,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len, num_heads = a.shape
    g = torch.empty(1, seq_len, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(
        1, seq_len, num_heads, dtype=torch.float32, device=b.device
    )
    grid = (triton.cdiv(seq_len, chunk_size), triton.cdiv(num_heads, 8))
    fused_gdn_gating_chunk_cumsum_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        beta,
        threshold,
        chunk_size,
        8,
        num_warps=2,
        num_stages=3,
    )
    return g, beta_output
