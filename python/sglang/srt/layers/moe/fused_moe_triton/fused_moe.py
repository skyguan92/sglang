# NOTE: this file will be separated into sglang/srt/layers/moe/moe_runner/triton_utils.py
# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/fused_moe.py

"""Fused MoE kernel."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.utils import get_moe_padding_size
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_xpu,
    use_intel_xpu_backend,
)
from sglang.srt.utils.custom_op import register_custom_op

from .fused_moe_triton_config import get_config_dtype_str, try_get_optimal_moe_config
from .fused_moe_triton_kernels import (
    act_and_mul_triton,
    invoke_fused_moe_kernel,
    moe_sum_reduce_triton,
    support_tensor_descriptor,
)
from .moe_align_block_size import moe_align_block_size

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import StandardTopKOutput

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_disable_aiter_moe_sum = (
    get_bool_env_var("UNIFYINFER_DISABLE_AITER_MOE_SUM") and _is_hip
)
_use_aiter_moe_sum = _use_aiter and not _disable_aiter_moe_sum
_is_xpu = is_xpu()
_use_sgl_xpu = use_intel_xpu_backend()
_use_unifyinfer_hip_mmq_port = (
    get_bool_env_var("UNIFYINFER_FUSED_MOE_HIP_MMQ_PORT") and _is_hip
)
_use_unifyinfer_mixed_tail_w13 = (
    get_bool_env_var("UNIFYINFER_EXPERIMENTAL_MOE_MIXED_TAIL_W13") and _is_hip
)
_use_unifyinfer_mixed_tail_w13_no_item = (
    get_bool_env_var("UNIFYINFER_EXPERIMENTAL_MOE_MIXED_TAIL_W13_NO_ITEM")
    and _is_hip
)


if _is_cuda:
    from sgl_kernel import gelu_and_mul, moe_sum_reduce, silu_and_mul
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_hip:
    from sgl_kernel import gelu_and_mul, silu_and_mul

    if _use_aiter_moe_sum:
        try:
            from aiter import moe_sum
        except ImportError:
            raise ImportError("aiter is required when SGLANG_USE_AITER is set to True")
    # Note: vllm_ops is not needed for HIP when _use_aiter=False
    # because the code uses moe_sum_reduce_triton as fallback (line 619)
elif _is_xpu:
    from sgl_kernel import moe_sum_reduce, silu_and_mul

# Try to import vllm_ops for non-CUDA/HIP/XPU platforms
_has_vllm_ops = False
if not _is_cuda and not _is_hip and not _is_xpu:
    try:
        from vllm import _custom_ops as vllm_ops

        _has_vllm_ops = True
    except ImportError:
        # Fallback: vllm not available, will use native PyTorch implementations
        _has_vllm_ops = False

padding_size = get_moe_padding_size(_use_aiter)


@register_custom_op(mutates_args=["hidden_states"])
def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: str = "silu",
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
    w1_prepacked: Optional[torch.Tensor] = None,
) -> None:
    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        b1,
        b2,
        True,
        activation,
        is_gated,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        False,
        routed_scaling_factor,
        gemm1_alpha,
        gemm1_limit,
        filter_expert,
        w1_prepacked,
    )


@register_custom_op(out_shape="hidden_states")
def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: str = "silu",
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
    w1_prepacked: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        b1,
        b2,
        False,
        activation,
        is_gated,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        no_combine=no_combine,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        filter_expert=filter_expert,
        w1_prepacked=w1_prepacked,
    )


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: StandardTopKOutput,
    moe_runner_config: MoeRunnerConfig,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    w1_prepacked: Optional[torch.Tensor] = None,
):
    topk_weights, topk_ids, _ = topk_output
    filter_expert = (
        moe_runner_config.num_experts is None
        or moe_runner_config.num_experts != moe_runner_config.num_local_experts
    )
    use_unifyinfer_mixed_tail_w13 = _should_use_unifyinfer_mixed_tail_w13(
        hidden_states=hidden_states,
        w1=w1,
        w1_prepacked=w1_prepacked,
        b1=b1,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        w1_scale=w1_scale,
        w1_zp=w1_zp,
        a1_scale=a1_scale,
        block_shape=block_shape,
    )
    if moe_runner_config.inplace:
        assert not moe_runner_config.no_combine, "no combine + inplace makes no sense"
        if use_unifyinfer_mixed_tail_w13:
            fused_experts_impl(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                b1,
                b2,
                True,
                moe_runner_config.activation,
                moe_runner_config.is_gated,
                moe_runner_config.apply_router_weight_on_input,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                use_int4_w4a16,
                per_channel_quant,
                w1_scale,
                w2_scale,
                w1_zp,
                w2_zp,
                a1_scale,
                a2_scale,
                block_shape,
                False,
                moe_runner_config.routed_scaling_factor,
                moe_runner_config.gemm1_alpha,
                moe_runner_config.gemm1_clamp_limit,
                filter_expert,
                w1_prepacked=w1_prepacked,
            )
            return hidden_states
        inplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            b1,
            b2,
            moe_runner_config.activation,
            moe_runner_config.is_gated,
            moe_runner_config.apply_router_weight_on_input,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            block_shape,
            moe_runner_config.routed_scaling_factor,
            moe_runner_config.gemm1_alpha,
            moe_runner_config.gemm1_clamp_limit,
            filter_expert,
            w1_prepacked,
        )
        return hidden_states
    else:
        if use_unifyinfer_mixed_tail_w13:
            return fused_experts_impl(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                b1,
                b2,
                False,
                moe_runner_config.activation,
                moe_runner_config.is_gated,
                moe_runner_config.apply_router_weight_on_input,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                use_int4_w4a16,
                per_channel_quant,
                w1_scale,
                w2_scale,
                w1_zp,
                w2_zp,
                a1_scale,
                a2_scale,
                block_shape,
                moe_runner_config.no_combine,
                moe_runner_config.routed_scaling_factor,
                moe_runner_config.gemm1_alpha,
                moe_runner_config.gemm1_clamp_limit,
                filter_expert,
                w1_prepacked=w1_prepacked,
            )
        return outplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            b1,
            b2,
            moe_runner_config.activation,
            moe_runner_config.is_gated,
            moe_runner_config.apply_router_weight_on_input,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            block_shape,
            no_combine=moe_runner_config.no_combine,
            routed_scaling_factor=moe_runner_config.routed_scaling_factor,
            gemm1_alpha=moe_runner_config.gemm1_alpha,
            gemm1_limit=moe_runner_config.gemm1_clamp_limit,
            filter_expert=filter_expert,
            w1_prepacked=w1_prepacked,
        )


@torch.compile
def moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)


@torch.compile
def _swiglu_silu_clamp_mul(x, gemm1_limit):
    gate, up = x.chunk(2, dim=-1)
    gate = F.silu(gate)
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * up


@torch.compile
def _swiglu_gpt_oss_sigmoid_alpha(x, gemm1_alpha, gemm1_limit):
    # NOTE: This variant uses gemm1_alpha, unlike _swiglu_silu_clamp_mul.
    # At present, only GPT-OSS uses this variant.
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)


@functools.lru_cache()
def _down_moe_use_tma():
    return support_tensor_descriptor()


@triton.jit
def _grouped_tail_direct_weight_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    starts_ptr,
    offs_ptr,
    expert_ids_ptr,
    total_groups,
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_group = pid // num_pid_n
    pid_n = pid % num_pid_n

    if pid_group >= total_groups:
        return

    start = tl.load(starts_ptr + pid_group).to(tl.int32)
    end = tl.load(offs_ptr + pid_group).to(tl.int32)
    group_rows = end - start
    if group_rows <= 0:
        return

    expert_id = tl.load(expert_ids_ptr + pid_group).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k_start * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a = tl.load(
            a_ptr + (start + offs_m)[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < group_rows) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + expert_id * stride_be
            + offs_n[None, :] * stride_bn
            + offs_k[:, None] * stride_bk,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(a, b)

    tl.store(
        c_ptr + (start + offs_m)[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < group_rows) & (offs_n[None, :] < N),
    )


def _should_use_unifyinfer_mixed_tail_w13(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_prepacked: Optional[torch.Tensor],
    b1: Optional[torch.Tensor],
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    w1_scale: Optional[torch.Tensor],
    w1_zp: Optional[torch.Tensor],
    a1_scale: Optional[torch.Tensor],
    block_shape: Optional[List[int]],
) -> bool:
    return (
        _use_unifyinfer_mixed_tail_w13
        and w1_prepacked is not None
        and b1 is None
        and hidden_states.dtype == torch.bfloat16
        and w1.dtype == torch.bfloat16
        and w1_prepacked.dtype == torch.bfloat16
        and not use_fp8_w8a8
        and not use_int8_w8a8
        and not use_int8_w8a16
        and not use_int4_w4a16
        and not per_channel_quant
        and w1_scale is None
        and w1_zp is None
        and a1_scale is None
        and block_shape is None
        and w1_prepacked.shape
        == (w1.shape[0], w1.shape[2], w1.shape[1])
    )


def _is_unifyinfer_single_storage_w13_alias(
    w1: torch.Tensor,
    w1_prepacked: Optional[torch.Tensor],
) -> bool:
    return (
        w1_prepacked is not None
        and not w1.is_contiguous()
        and w1_prepacked.is_contiguous()
        and w1.dtype == torch.bfloat16
        and w1_prepacked.dtype == torch.bfloat16
        and w1.data_ptr() == w1_prepacked.data_ptr()
        and w1.shape == (w1_prepacked.shape[0], w1_prepacked.shape[2], w1_prepacked.shape[1])
        and w1.stride(0) == w1_prepacked.stride(0)
        and w1.stride(1) == w1_prepacked.stride(2)
        and w1.stride(2) == w1_prepacked.stride(1)
    )


def _build_unifyinfer_mixed_tail_metadata(
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    block_size_m: int,
    num_valid_assignments: int,
) -> dict[str, torch.Tensor]:
    device = sorted_token_ids.device
    if _use_unifyinfer_mixed_tail_w13_no_item:
        num_tokens_post_padded_value = (
            sorted_token_ids.numel() // block_size_m
        ) * block_size_m
    else:
        num_tokens_post_padded_value = int(num_tokens_post_padded.item())
    if num_tokens_post_padded_value <= 0:
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        return {
            "full_sorted_token_ids": empty_i32,
            "full_expert_ids": empty_i32,
            "full_num_tokens_post_padded": torch.zeros(
                (1,), dtype=torch.int32, device=device
            ),
            "tail_assignment_ids": empty_i32,
            "tail_starts": empty_i32,
            "tail_ends": empty_i32,
            "tail_expert_ids": empty_i32,
            "zero_tail_assignment_ids": empty_i32,
        }

    num_blocks = triton.cdiv(num_tokens_post_padded_value, block_size_m)
    sorted_blocks = sorted_token_ids[: num_blocks * block_size_m].view(
        num_blocks, block_size_m
    )
    expert_blocks = expert_ids[:num_blocks]
    valid_mask = sorted_blocks < num_valid_assignments
    valid_counts = valid_mask.sum(dim=1)

    full_block_mask = valid_counts == block_size_m
    tail_block_mask = (
        (valid_counts > 0) & (valid_counts < block_size_m) & (expert_blocks != -1)
    )
    zero_tail_block_mask = (
        (valid_counts > 0) & (valid_counts < block_size_m) & (expert_blocks == -1)
    )

    full_sorted_token_ids = sorted_blocks[full_block_mask].reshape(-1).contiguous()
    full_expert_ids = expert_blocks[full_block_mask].contiguous()
    full_num_tokens_post_padded = torch.tensor(
        [full_sorted_token_ids.numel()],
        dtype=torch.int32,
        device=device,
    )

    tail_assignment_ids = sorted_blocks[tail_block_mask][
        valid_mask[tail_block_mask]
    ].contiguous()
    tail_expert_ids = expert_blocks[tail_block_mask].contiguous()
    tail_counts = valid_counts[tail_block_mask].to(dtype=torch.int32)
    if tail_counts.numel() > 0:
        tail_ends = tail_counts.cumsum(dim=0)
        tail_starts = torch.cat(
            [
                torch.zeros((1,), dtype=torch.int32, device=device),
                tail_ends[:-1],
            ]
        )
    else:
        tail_starts = torch.empty((0,), dtype=torch.int32, device=device)
        tail_ends = torch.empty((0,), dtype=torch.int32, device=device)

    zero_tail_assignment_ids = sorted_blocks[zero_tail_block_mask][
        valid_mask[zero_tail_block_mask]
    ].contiguous()

    return {
        "full_sorted_token_ids": full_sorted_token_ids,
        "full_expert_ids": full_expert_ids,
        "full_num_tokens_post_padded": full_num_tokens_post_padded,
        "tail_assignment_ids": tail_assignment_ids,
        "tail_starts": tail_starts,
        "tail_ends": tail_ends,
        "tail_expert_ids": tail_expert_ids,
        "zero_tail_assignment_ids": zero_tail_assignment_ids,
    }


def _invoke_unifyinfer_mixed_tail_w13(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_prepacked: torch.Tensor,
    intermediate_cache1: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
    compute_type: tl.dtype,
    filter_expert: bool,
) -> None:
    block_size_m = int(config["BLOCK_SIZE_M"])
    metadata = _build_unifyinfer_mixed_tail_metadata(
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        block_size_m=block_size_m,
        num_valid_assignments=topk_ids.numel(),
    )

    full_num_tokens_post_padded = metadata["full_num_tokens_post_padded"]
    if _use_unifyinfer_mixed_tail_w13_no_item:
        has_full_blocks = metadata["full_sorted_token_ids"].numel() > 0
    else:
        has_full_blocks = int(full_num_tokens_post_padded.item()) > 0
    if has_full_blocks:
        invoke_fused_moe_kernel(
            hidden_states,
            w1,
            None,
            intermediate_cache1,
            None,
            None,
            None,
            topk_weights,
            topk_ids,
            metadata["full_sorted_token_ids"],
            metadata["full_expert_ids"],
            full_num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
            filter_expert=filter_expert,
        )

    zero_tail_assignment_ids = metadata["zero_tail_assignment_ids"]
    if zero_tail_assignment_ids.numel() > 0:
        zero_tail_assignment_ids = zero_tail_assignment_ids.to(dtype=torch.long)
        intermediate_cache1.index_copy_(
            0,
            zero_tail_assignment_ids,
            torch.zeros(
                (zero_tail_assignment_ids.numel(), intermediate_cache1.shape[1]),
                device=intermediate_cache1.device,
                dtype=intermediate_cache1.dtype,
            ),
        )

    tail_assignment_ids = metadata["tail_assignment_ids"]
    if tail_assignment_ids.numel() == 0:
        return

    tail_assignment_ids_long = tail_assignment_ids.to(dtype=torch.long)
    tail_token_ids = torch.div(
        tail_assignment_ids_long, top_k, rounding_mode="floor"
    ).contiguous()
    tail_hidden_states = hidden_states.index_select(0, tail_token_ids).contiguous()
    if mul_routed_weight:
        tail_router_weights = topk_weights.reshape(-1).index_select(
            0, tail_assignment_ids_long
        )
        tail_hidden_states.mul_(tail_router_weights.to(tail_hidden_states.dtype).unsqueeze(1))

    tail_out = torch.empty(
        (tail_assignment_ids_long.numel(), w1.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    block_size_n = max(int(config["BLOCK_SIZE_N"]), 64)
    block_size_k = 64
    if hidden_states.shape[1] % block_size_k != 0:
        raise ValueError("mixed-tail grouped kernel requires K to be divisible by 64")

    _grouped_tail_direct_weight_kernel[
        lambda meta: (
            metadata["tail_expert_ids"].numel()
            * triton.cdiv(w1.shape[1], meta["BLOCK_SIZE_N"]),
        )
    ](
        tail_hidden_states,
        w1_prepacked,
        tail_out,
        metadata["tail_starts"],
        metadata["tail_ends"],
        metadata["tail_expert_ids"],
        metadata["tail_expert_ids"].numel(),
        w1.shape[1],
        hidden_states.shape[1],
        tail_hidden_states.stride(0),
        tail_hidden_states.stride(1),
        w1_prepacked.stride(0),
        w1_prepacked.stride(2),
        w1_prepacked.stride(1),
        tail_out.stride(0),
        tail_out.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        num_warps=4,
        num_stages=2,
    )
    intermediate_cache1.index_copy_(0, tail_assignment_ids_long, tail_out)


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    inplace: bool = False,
    activation: str = "silu",
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
    w1_prepacked: Optional[torch.Tensor] = None,
):
    padded_size = padding_size
    if not (use_fp8_w8a8 or use_int8_w8a8) or block_shape is not None or _use_aiter:
        padded_size = 0

    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.shape[1] // 2 == w1.shape[2], "Hidden size mismatch"
    else:
        assert (
            hidden_states.shape[1] == w1.shape[2] - padded_size
        ), f"Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    alias_ok = _is_unifyinfer_single_storage_w13_alias(w1, w1_prepacked)
    assert (
        w1.is_contiguous() or alias_ok
    ), "Expert weights1 must be contiguous or a supported single-storage W13 alias"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = 64 * 1024
    M = min(num_tokens, CHUNK_SIZE)
    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        dtype=hidden_states.dtype,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
        config_dtype,
        block_shape=block_shape,
        per_channel_quant=per_channel_quant,
        return_down_config=True,
    )

    config, (down_config, max_block_m) = get_config_func(M)
    down_moe_use_tma = (
        _down_moe_use_tma()
        and down_config is not None
        and down_config.pop("USE_TMA", False)
    )
    topk = topk_ids.shape[1]
    max_padded_tokens = (
        min(M * topk, E + 1) * (max_block_m - 1) if down_moe_use_tma else 0
    )
    total_tokens = M * topk + max_padded_tokens
    cache = torch.empty(
        total_tokens * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk * w2.shape[1]].view(
        (M, topk, w2.shape[1]),
    )

    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            config, (down_config, _) = get_config_func(tokens_in_chunk)
            down_moe_use_tma = (
                _down_moe_use_tma()
                and down_config is not None
                and down_config.pop("USE_TMA", False)
            )
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        padded_tokens = (
            min(tokens_in_chunk * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1)
            if down_moe_use_tma
            else 0
        )
        total_tokens = tokens_in_chunk * topk + padded_tokens
        intermediate_cache1 = cache[: total_tokens * N].view(
            (total_tokens, N),
        )
        intermediate_cache2 = torch.empty(
            (total_tokens, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        use_fused_moe_sum_all_reduce = (
            get_global_server_args().enable_fused_moe_sum_all_reduce
            and (not no_combine)
            and (curr_topk_ids.shape[1] > 2)
            and (not use_int8_w8a16)
            and (not use_int4_w4a16)
        )

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids, config["BLOCK_SIZE_M"], E
        )

        if _should_use_unifyinfer_mixed_tail_w13(
            hidden_states=curr_hidden_states,
            w1=w1,
            w1_prepacked=w1_prepacked,
            b1=b1,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            w1_scale=w1_scale,
            w1_zp=w1_zp,
            a1_scale=a1_scale,
            block_shape=block_shape,
        ):
            _invoke_unifyinfer_mixed_tail_w13(
                hidden_states=curr_hidden_states,
                w1=w1,
                w1_prepacked=w1_prepacked,
                intermediate_cache1=intermediate_cache1,
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=apply_router_weight_on_input,
                top_k=topk_ids.shape[1],
                config=config,
                compute_type=compute_type,
                filter_expert=filter_expert,
            )
        else:
            invoke_fused_moe_kernel(
                curr_hidden_states,
                w1,
                b1,
                intermediate_cache1,
                a1_scale,
                w1_scale,
                w1_zp,
                curr_topk_weights,
                curr_topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                apply_router_weight_on_input,
                topk_ids.shape[1],
                config,
                compute_type=compute_type,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                per_channel_quant=per_channel_quant,
                block_shape=block_shape,
                c_sorted=down_moe_use_tma,
                filter_expert=filter_expert,
            )

        # Activation function with multiplication
        if activation == "silu" and is_gated:
            # - gemm1_alpha != None: GPT-OSS-style swiglu(alpha, limit)
            # - gemm1_alpha == None and gemm1_limit != None: silu+clamp+mul(limit-only)
            if gemm1_alpha is not None:
                assert gemm1_limit is not None
                intermediate_cache2 = _swiglu_gpt_oss_sigmoid_alpha(
                    intermediate_cache1.view(-1, N), gemm1_alpha, gemm1_limit
                )
            elif gemm1_limit is not None:
                intermediate_cache2 = _swiglu_silu_clamp_mul(
                    intermediate_cache1.view(-1, N), gemm1_limit
                )
            elif _is_cuda or _is_hip or _is_xpu:
                use_triton_activation = filter_expert or _use_unifyinfer_hip_mmq_port
                if not use_triton_activation:
                    silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
                else:
                    act_and_mul_triton(
                        intermediate_cache1.view(-1, N),
                        intermediate_cache2,
                        config,
                        curr_topk_ids,
                        expert_ids,
                        down_moe_use_tma,
                        activation,
                    )
            else:
                if _has_vllm_ops:
                    vllm_ops.silu_and_mul(
                        intermediate_cache2, intermediate_cache1.view(-1, N)
                    )
                else:
                    # Fallback: native PyTorch silu_and_mul
                    x = intermediate_cache1.view(-1, N)
                    d = x.shape[-1] // 2
                    intermediate_cache2.copy_(F.silu(x[..., :d]) * x[..., d:])
        elif activation == "gelu" and is_gated:
            assert gemm1_alpha is None, "gemm1_alpha is not supported for gelu"
            assert gemm1_limit is None, "gemm1_limit is not supported for gelu"
            if _is_cuda or _is_hip:
                if not filter_expert:
                    gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
                else:
                    act_and_mul_triton(
                        intermediate_cache1.view(-1, N),
                        intermediate_cache2,
                        config,
                        curr_topk_ids,
                        expert_ids,
                        down_moe_use_tma,
                        activation,
                    )
            else:
                if _has_vllm_ops:
                    vllm_ops.gelu_and_mul(
                        intermediate_cache2, intermediate_cache1.view(-1, N)
                    )
                else:
                    # Fallback: native PyTorch gelu_and_mul
                    x = intermediate_cache1.view(-1, N)
                    d = x.shape[-1] // 2
                    intermediate_cache2.copy_(F.gelu(x[..., :d]) * x[..., d:])
        # Activation function without multiplication
        elif activation == "silu" and not is_gated:
            intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))
        elif activation == "gelu" and not is_gated:
            intermediate_cache2 = F.gelu(intermediate_cache1.view(-1, N))
        elif activation == "relu2" and not is_gated:
            intermediate_cache2 = torch.square(F.relu(intermediate_cache1.view(-1, N)))
        else:
            raise ValueError(f"Unsupported activation: {activation=}, with {is_gated=}")

        out_slice = None
        if use_fused_moe_sum_all_reduce:
            out_slice = out_hidden_states[begin_chunk_idx:end_chunk_idx]
            out_slice.zero_()

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            b2,
            (
                out_slice
                if use_fused_moe_sum_all_reduce
                else (
                    intermediate_cache3
                    if not no_combine and topk_ids.shape[1] != 1
                    else out_hidden_states[begin_chunk_idx:end_chunk_idx].unsqueeze(0)
                )
            ),
            a2_scale,
            w2_scale,
            w2_zp,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            down_config or config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            a_use_tma=down_moe_use_tma,
            b_use_tma=down_moe_use_tma,
            filter_expert=filter_expert,
            fuse_sum_all_reduce=use_fused_moe_sum_all_reduce,
            router_topk=curr_topk_ids.shape[1],
        )

        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if no_combine:
            pass
        elif _is_cuda:
            if use_fused_moe_sum_all_reduce:
                if routed_scaling_factor is None:
                    routed_scaling_factor = 1.0
                if routed_scaling_factor != 1.0:
                    assert out_slice is not None
                    out_slice.mul_(routed_scaling_factor)
            elif topk_ids.shape[1] == 1 and routed_scaling_factor == 1.0:
                pass  # we write directly into out_hidden_states
            elif topk_ids.shape[1] == 2 and routed_scaling_factor == 1.0:
                torch.add(
                    intermediate_cache3[:, 0],
                    intermediate_cache3[:, 1],
                    out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
                ).squeeze(dim=1)
            else:
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if tokens_in_chunk <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )

        elif _is_hip:
            if _use_aiter_moe_sum:
                moe_sum(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                )
            else:
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if tokens_in_chunk <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce_triton(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
        elif _is_xpu:
            moe_sum_reduce(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states[begin_chunk_idx:end_chunk_idx],
                routed_scaling_factor,
            )
        else:
            if _has_vllm_ops:
                vllm_ops.moe_sum(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                )
            else:
                # Fallback: use triton moe_sum_reduce when vllm is not available
                moe_sum_reduce_triton(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                    routed_scaling_factor,
                )

    return out_hidden_states


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: StandardTopKOutput,
    moe_runner_config: MoeRunnerConfig = MoeRunnerConfig(),
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - topk_output (StandardTopKOutput): The top-k output of the experts.
    - moe_runner_config (MoeRunnerConfig): The configuration for the MoE runner.
    - b1 (Optional[torch.Tensor]): Optional bias for w1.
    - b2 (Optional[torch.Tensor]): Optional bias for w2.
    - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a8 (bool): If True, use int8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a16 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int4_w4a16 (bool): If True, use matmul of int4 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - a1_scale (Optional[torch.Tensor]): Optional scale to be used for
        a1.
    - a2_scale (Optional[torch.Tensor]): Optional scale to be used for
        a2.
    - block_shape: (Optional[List[int]]): Optional block size for block-wise
        quantization.
    - gemm1_alpha (Optional[float]): Optional gemm1_alpha for the activation
        function.
    - gemm1_limit (Optional[float]): Optional gemm1_limit for the swiglu activation
        function.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    if _use_sgl_xpu:
        topk_weight, topk_ids, _ = topk_output
        from sgl_kernel import fused_experts as sgl_fused_experts

        return sgl_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            b1=b1,
            b2=b2,
            use_fp8_w8a8=use_fp8_w8a8,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
        )

    return fused_experts(
        hidden_states,
        w1,
        w2,
        topk_output,
        moe_runner_config=moe_runner_config,
        b1=b1,
        b2=b2,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
    )
