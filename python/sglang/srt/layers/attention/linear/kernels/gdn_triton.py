import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import get_bool_env_var, get_int_env_var, is_cpu, is_npu

_is_cpu = is_cpu()
_is_npu = is_npu()

if not _is_cpu:
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
    from sglang.srt.layers.attention.linear.kernels.gdn_ggml_row import (
        can_use_ggml_gdn_row_update,
        ggml_gdn_row_update,
    )
    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_update,
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

if _is_npu:
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update_npu,
    )

    chunk_gated_delta_rule = chunk_gated_delta_rule_npu
    fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu
elif _is_cpu:
    from sgl_kernel.mamba import chunk_gated_delta_rule_cpu

    chunk_gated_delta_rule = chunk_gated_delta_rule_cpu
    fused_sigmoid_gating_delta_rule_update = (
        torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update_cpu
    )

_use_unifyinfer_qwen35_gdn_recurrent_extend = get_bool_env_var(
    "UNIFYINFER_QWEN35_GDN_RECURRENT_EXTEND"
)
_unifyinfer_qwen35_gdn_recurrent_extend_min_tokens = get_int_env_var(
    "UNIFYINFER_QWEN35_GDN_RECURRENT_EXTEND_MIN_TOKENS"
)
_unifyinfer_qwen35_gdn_recurrent_trace_shape = get_bool_env_var(
    "UNIFYINFER_QWEN35_GDN_RECURRENT_TRACE_SHAPE"
)
_unifyinfer_qwen35_gdn_recurrent_trace_shape_limit = get_int_env_var(
    "UNIFYINFER_QWEN35_GDN_RECURRENT_TRACE_SHAPE_LIMIT", 16
)
_unifyinfer_qwen35_gdn_recurrent_trace_shape_count = 0
_use_unifyinfer_qwen35_gdn_ggml_row_kernel = get_bool_env_var(
    "UNIFYINFER_QWEN35_GDN_GGML_ROW_KERNEL"
)
_unifyinfer_qwen35_gdn_ggml_row_kernel_min_tokens = get_int_env_var(
    "UNIFYINFER_QWEN35_GDN_GGML_ROW_KERNEL_MIN_TOKENS", 128
)


def _maybe_trace_recurrent_extend_shape(
    *,
    branch: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    has_mamba_track_mask: bool,
) -> None:
    global _unifyinfer_qwen35_gdn_recurrent_trace_shape_count
    if not _unifyinfer_qwen35_gdn_recurrent_trace_shape:
        return
    if (
        _unifyinfer_qwen35_gdn_recurrent_trace_shape_count
        >= _unifyinfer_qwen35_gdn_recurrent_trace_shape_limit
    ):
        return
    _unifyinfer_qwen35_gdn_recurrent_trace_shape_count += 1

    query_start_loc_preview = None
    if query_start_loc is not None:
        query_start_loc_preview = query_start_loc.detach().cpu().tolist()[:8]
    cache_indices_preview = cache_indices.detach().cpu().tolist()[:8]
    print(
        "[unifyinfer-qwen35-gdn-recurrent-shape] "
        f"branch={branch} "
        f"q_shape={tuple(q.shape)} "
        f"k_shape={tuple(k.shape)} "
        f"v_shape={tuple(v.shape)} "
        f"g_shape={tuple(g.shape)} "
        f"beta_shape={tuple(beta.shape)} "
        f"min_tokens={_unifyinfer_qwen35_gdn_recurrent_extend_min_tokens} "
        f"cache_indices_numel={cache_indices.numel()} "
        f"cache_indices={cache_indices_preview} "
        f"query_start_loc_numel="
        f"{query_start_loc.numel() if query_start_loc is not None else None} "
        f"query_start_loc={query_start_loc_preview} "
        f"has_mamba_track_mask={has_mamba_track_mask}",
        flush=True,
    )


class TritonGDNKernel(LinearAttnKernelBase):
    """Triton-based kernel for GDN (Gated Delta Network) linear attention."""

    supports_packed_decode: bool = not _is_cpu and not _is_npu

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> torch.Tensor:
        """Packed decode fast path: fuse QKV extraction + gating + recurrent
        update into a single Triton kernel, eliminating intermediate tensors
        and extra kernel launches.

        Args:
            mixed_qkv: [B, qkv_dim] packed projection output after conv1d.
            a, b: [B, HV] gating inputs.
            A_log: [HV] log-space decay parameter.
            dt_bias: [HV] time-step bias.
            scale: attention scale factor (typically head_k_dim ** -0.5).
            ssm_states: [num_slots, HV, V, K] full state pool.
            cache_indices: [B] per-request state slot indices.
            num_v_heads: number of value heads (after TP sharding).
            head_v_dim: dimension per value head.

        Returns:
            output tensor of shape [1, B, HV, V] matching the existing
            decode kernel output layout.
        """
        B = mixed_qkv.shape[0]
        # Packed kernel expects output shape [B, 1, HV, V]
        out = mixed_qkv.new_empty(B, 1, num_v_heads, head_v_dim)

        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
        )

        # Convert [B, 1, HV, V] → [1, B, HV, V] to match existing output
        # layout. transpose() returns a view — zero cost.
        return out.transpose(0, 1)

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        use_recurrent_extend = False
        has_mamba_track_mask = False
        if (
            _use_unifyinfer_qwen35_gdn_recurrent_extend
            and q.shape[1] >= _unifyinfer_qwen35_gdn_recurrent_extend_min_tokens
        ):
            has_mamba_track_mask = kwargs.get("has_mamba_track_mask", False)
            use_recurrent_extend = (
                not (_is_cpu or _is_npu)
                and not has_mamba_track_mask
                and q.shape[0] == 1
                and cache_indices.numel() == 1
                and query_start_loc is not None
                and query_start_loc.numel() == 2
            )
        elif _unifyinfer_qwen35_gdn_recurrent_trace_shape:
            has_mamba_track_mask = kwargs.get("has_mamba_track_mask", False)

        if _unifyinfer_qwen35_gdn_recurrent_trace_shape:
            _maybe_trace_recurrent_extend_shape(
                branch="recurrent" if use_recurrent_extend else "chunk",
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                has_mamba_track_mask=has_mamba_track_mask,
            )
        if use_recurrent_extend:
            return self.extend_recurrent(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
            )

        return self.extend_chunk(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            g_is_chunk_cumsum=kwargs.get("g_is_chunk_cumsum", False),
        )

    def extend_recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        **kwargs,
    ) -> tuple:
        out = fused_recurrent_gated_delta_rule_update(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=k.shape[-1] ** -0.5,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=None,
            use_qk_l2norm_in_kernel=True,
        )
        return out, None, None

    def extend_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        has_mamba_track_mask = kwargs.get("has_mamba_track_mask", False)
        g_is_chunk_cumsum = kwargs.get("g_is_chunk_cumsum", False)
        if (
            _use_unifyinfer_qwen35_gdn_ggml_row_kernel
            and not (_is_cpu or _is_npu)
            and can_use_ggml_gdn_row_update(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=ssm_states,
                initial_state_indices=cache_indices,
                min_tokens=_unifyinfer_qwen35_gdn_ggml_row_kernel_min_tokens,
                has_mamba_track_mask=has_mamba_track_mask,
                g_is_chunk_cumsum=g_is_chunk_cumsum,
            )
        ):
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            g = g.contiguous()
            beta = beta.contiguous()
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)
            out = ggml_gdn_row_update(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=ssm_states,
                initial_state_indices=cache_indices,
                scale=k.shape[-1] ** -0.5,
            )
            return out, None, None

        recurrent_state = ssm_states
        recurrent_state_indices_args = {"initial_state_indices": cache_indices}
        if _is_npu or _is_cpu:
            recurrent_state = ssm_states[cache_indices]
            recurrent_state_indices_args = {}
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            cu_seqlens=query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            g_is_chunk_cumsum=g_is_chunk_cumsum,
            **recurrent_state_indices_args,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,
            # target_verify specific parameters
            disable_state_update=True,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
        )
