import hashlib
import json
import os
from contextlib import nullcontext
from typing import Optional, Tuple, Union

import torch
from torch.autograd.profiler import record_function

from sglang.srt.layers.attention.fla.fused_gdn_gating import (
    fused_gdn_gating,
    fused_gdn_gating_chunk_cumsum,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import get_bool_env_var, get_int_env_var, is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    fused_gdn_gating = fused_gdn_gating_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu

_use_unifyinfer_qwen35_trace_attn_split = get_bool_env_var(
    "UNIFYINFER_QWEN35_TRACE_ATTN_SPLIT"
)
_use_unifyinfer_qwen35_dflash_boundary_digests = get_bool_env_var(
    "UNIFYINFER_QWEN35_DFLASH_BOUNDARY_DIGESTS"
)
_use_unifyinfer_qwen35_gdn_fused_gate_cumsum = (
    get_bool_env_var("UNIFYINFER_QWEN35_GDN_FUSED_GATE_CUMSUM")
    and get_bool_env_var("UNIFYINFER_QWEN35_FLA_DIRECT_EXTEND_NOAUTOGRAD")
    and get_bool_env_var("UNIFYINFER_QWEN35_FLA_SINGLE_SEQ_NOVARLEN_EXTEND")
    and not is_cpu()
    and not is_npu()
)
_use_unifyinfer_qwen35_gdn_recurrent_backend_select = (
    get_bool_env_var("UNIFYINFER_QWEN35_GDN_RECURRENT_BACKEND_SELECT")
    and not is_cpu()
    and not is_npu()
)
_unifyinfer_qwen35_gdn_recurrent_backend_select_min_tokens = get_int_env_var(
    "UNIFYINFER_QWEN35_GDN_RECURRENT_EXTEND_MIN_TOKENS"
)


def _qwen35_gdn_trace_span(name: str):
    if not _use_unifyinfer_qwen35_trace_attn_split:
        return nullcontext()
    return record_function(name)


def _qwen35_gdn_boundary_max_rows() -> int:
    raw = os.environ.get("UNIFYINFER_QWEN35_DFLASH_BOUNDARY_DIGEST_MAX_ROWS", "32")
    try:
        return max(0, int(raw))
    except ValueError:
        return 32


def _qwen35_gdn_digest_tensor(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().to(device="cpu")
    original_dtype = str(cpu.dtype)
    raw_tensor = cpu.view(torch.uint16) if cpu.dtype == torch.bfloat16 else cpu
    try:
        raw_bytes = raw_tensor.numpy().tobytes()
        digest_dtype = original_dtype
    except Exception:
        raw_tensor = cpu.to(torch.float32)
        raw_bytes = raw_tensor.numpy().tobytes()
        digest_dtype = "torch.float32_from_" + original_dtype
    hasher = hashlib.sha256()
    hasher.update(original_dtype.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(digest_dtype.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(raw_bytes)
    return "sha256:" + hasher.hexdigest()


def _qwen35_gdn_record_boundary_digest(
    forward_batch: Optional[ForwardBatch],
    *,
    layer_id: int,
    stage: str,
    tensor: Optional[torch.Tensor],
) -> None:
    if (
        not _use_unifyinfer_qwen35_dflash_boundary_digests
        or forward_batch is None
        or not forward_batch.forward_mode.is_target_verify()
        or tensor is None
        or not isinstance(tensor, torch.Tensor)
        or tensor.ndim == 0
    ):
        return
    probe_layers = getattr(
        forward_batch,
        "_unifyinfer_qwen35_dflash_boundary_probe_layers",
        None,
    )
    if not isinstance(probe_layers, set) or int(layer_id) not in probe_layers:
        return

    max_rows = _qwen35_gdn_boundary_max_rows()
    if max_rows <= 0:
        return
    view = tensor.detach().reshape(int(tensor.shape[0]), -1)
    rows = min(int(view.shape[0]), max_rows)
    records = getattr(
        forward_batch,
        "_unifyinfer_qwen35_dflash_boundary_digests",
        None,
    )
    if records is None:
        records = []
        setattr(forward_batch, "_unifyinfer_qwen35_dflash_boundary_digests", records)
    records.append(
        {
            "layer_id": int(layer_id),
            "stage": str(stage),
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": str(tensor.dtype),
            "rows_recorded": rows,
            "row_digests": [
                _qwen35_gdn_digest_tensor(view[row_index])
                for row_index in range(rows)
            ],
        }
    )


def _qwen35_gdn_index_rows(
    tensor: Optional[torch.Tensor],
    indices: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None
    if indices is None or not isinstance(indices, torch.Tensor) or indices.numel() == 0:
        return None
    return tensor.index_select(0, indices.detach().to(device=tensor.device, dtype=torch.long))


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonGDNKernel()

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            self.decode_kernel = CuteDSLGDNKernel()
        elif decode_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            flashinfer_kernel = FlashInferGDNKernel()
            self.decode_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            raise ValueError(
                "CuTe DSL backend only supports decode, not prefill. "
                "Use --linear-attn-prefill-backend triton instead."
            )
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            # Reuse the FlashInfer kernel if already created for decode
            if decode_backend.is_flashinfer():
                self.extend_kernel = flashinfer_kernel
            else:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    FlashInferGDNKernel,
                )

                flashinfer_kernel = FlashInferGDNKernel()
                self.extend_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        # Verify kernel: use FlashInfer if either decode or prefill selected it
        if decode_backend.is_flashinfer() or prefill_backend.is_flashinfer():
            self.verify_kernel = flashinfer_kernel
        else:
            self.verify_kernel = triton_kernel

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
        )

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
    ) -> Optional[torch.Tensor]:
        """Attempt packed decode. Returns output tensor or None if
        the decode kernel does not support packed decode."""
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )

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
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
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
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

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
        if not hasattr(self.extend_kernel, "extend_chunk"):
            return self.extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                **kwargs,
            )
        return self.extend_kernel.extend_chunk(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
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
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        if not hasattr(self.extend_kernel, "extend_recurrent"):
            return self.extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                **kwargs,
            )
        return self.extend_kernel.extend_recurrent(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            **kwargs,
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
        **kwargs,
    ) -> torch.Tensor:
        return self.verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        if not is_cpu() and not is_npu():
            assert (
                self.conv_states_shape[-1] < FLA_CHUNK_SIZE
            ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"

        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)
        self.verify_intermediate_state_indices = torch.arange(
            self.req_to_token_pool.size, dtype=torch.int32, device=model_runner.device
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if getattr(self.forward_metadata, "has_mamba_track_mask", False):
            self.forward_metadata.mamba_track_mask_indices = (
                forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
            )
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[
                    self.forward_metadata.mamba_track_mask_indices
                ]
            )

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        assert isinstance(mixed_qkv, torch.Tensor)
        with _qwen35_gdn_trace_span("_qwen35_gdncore_conv"):
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices,
            )

        # Skip split + reshape + separate gating kernel by consuming
        # the packed mixed_qkv directly in a single fused Triton kernel.
        if self.kernel_dispatcher.supports_packed_decode:
            with _qwen35_gdn_trace_span("_qwen35_gdncore_dispatch"):
                core_attn_out = self.kernel_dispatcher.packed_decode(
                    mixed_qkv=mixed_qkv,
                    a=a,
                    b=b,
                    A_log=layer.A_log,
                    dt_bias=layer.dt_bias,
                    scale=layer.head_k_dim**-0.5,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    num_v_heads=layer.num_v_heads,
                    head_v_dim=layer.head_v_dim,
                )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices
            )
            return core_attn_out

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        with _qwen35_gdn_trace_span("_qwen35_gdncore_dispatch"):
            core_attn_out = self.kernel_dispatcher.decode(
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )

        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            intermediate_state_indices = self.verify_intermediate_state_indices
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            target_cache_indices = cache_indices[:batch_size]
            target_intermediate_state_indices = intermediate_state_indices[:batch_size]
            _qwen35_gdn_record_boundary_digest(
                forward_batch,
                layer_id=int(layer.layer_id),
                stage="linear_attn_core_cache_indices",
                tensor=target_cache_indices,
            )
            _qwen35_gdn_record_boundary_digest(
                forward_batch,
                layer_id=int(layer.layer_id),
                stage="linear_attn_core_query_start_loc",
                tensor=query_start_loc,
            )
            _qwen35_gdn_record_boundary_digest(
                forward_batch,
                layer_id=int(layer.layer_id),
                stage="linear_attn_core_intermediate_state_indices",
                tensor=target_intermediate_state_indices,
            )
            _qwen35_gdn_record_boundary_digest(
                forward_batch,
                layer_id=int(layer.layer_id),
                stage="linear_attn_core_retrieve_next_token",
                tensor=retrieve_next_token,
            )
            _qwen35_gdn_record_boundary_digest(
                forward_batch,
                layer_id=int(layer.layer_id),
                stage="linear_attn_core_retrieve_next_sibling",
                tensor=retrieve_next_sibling,
            )
            _qwen35_gdn_record_boundary_digest(
                forward_batch,
                layer_id=int(layer.layer_id),
                stage="linear_attn_core_retrieve_parent_token",
                tensor=retrieve_parent_token,
            )
            _qwen35_gdn_record_boundary_digest(
                forward_batch,
                layer_id=int(layer.layer_id),
                stage="linear_attn_core_conv_state",
                tensor=_qwen35_gdn_index_rows(conv_states, target_cache_indices),
            )
            _qwen35_gdn_record_boundary_digest(
                forward_batch,
                layer_id=int(layer.layer_id),
                stage="linear_attn_core_initial_ssm_state",
                tensor=_qwen35_gdn_index_rows(ssm_states, target_cache_indices),
            )
            _qwen35_gdn_record_boundary_digest(
                forward_batch,
                layer_id=int(layer.layer_id),
                stage="linear_attn_core_intermediate_ssm_state",
                tensor=_qwen35_gdn_index_rows(
                    intermediate_state_cache,
                    target_intermediate_state_indices,
                ),
            )
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            with _qwen35_gdn_trace_span("_qwen35_gdncore_conv"):
                mixed_qkv_processed = causal_conv1d_update(
                    mixed_qkv_reshaped,
                    conv_states,
                    layer.conv_weights,
                    layer.bias,
                    layer.activation,
                    conv_state_indices=cache_indices[:batch_size],
                    intermediate_conv_window=intermediate_conv_window_cache,
                    intermediate_state_indices=intermediate_state_indices[:batch_size],
                    retrieve_next_token=retrieve_next_token,
                    retrieve_next_sibling=retrieve_next_sibling,
                    retrieve_parent_token=retrieve_parent_token,
                )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
            _qwen35_gdn_record_boundary_digest(
                forward_batch,
                layer_id=int(layer.layer_id),
                stage="linear_attn_core_conv_mixed_qkv",
                tensor=mixed_qkv,
            )
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if getattr(forward_metadata, "has_mamba_track_mask", False):
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )

            with _qwen35_gdn_trace_span("_qwen35_gdncore_conv"):
                mixed_qkv = causal_conv1d_fn(
                    mixed_qkv,
                    layer.conv_weights,
                    layer.bias,
                    activation=layer.activation,
                    conv_states=conv_states,
                    has_initial_state=has_initial_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                ).transpose(0, 1)[:seq_len]

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)
        if is_target_verify:
            _qwen35_gdn_record_boundary_digest(
                forward_batch,
                layer_id=int(layer.layer_id),
                stage="linear_attn_core_query",
                tensor=query.squeeze(0),
            )
            _qwen35_gdn_record_boundary_digest(
                forward_batch,
                layer_id=int(layer.layer_id),
                stage="linear_attn_core_key",
                tensor=key.squeeze(0),
            )
            _qwen35_gdn_record_boundary_digest(
                forward_batch,
                layer_id=int(layer.layer_id),
                stage="linear_attn_core_value",
                tensor=value.squeeze(0),
            )

        if is_target_verify:
            with _qwen35_gdn_trace_span("_qwen35_gdncore_dispatch"):
                core_attn_out = self.kernel_dispatcher.target_verify(
                    A_log=layer.A_log,
                    dt_bias=layer.dt_bias,
                    q=query,
                    k=key,
                    v=value,
                    a=a,
                    b=b,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    intermediate_states_buffer=intermediate_state_cache,
                    intermediate_state_indices=intermediate_state_indices,
                    cache_steps=forward_batch.spec_info.draft_token_num,
                    retrieve_parent_token=retrieve_parent_token,
                )
            _qwen35_gdn_record_boundary_digest(
                forward_batch,
                layer_id=int(layer.layer_id),
                stage="linear_attn_core_kernel_output",
                tensor=core_attn_out.squeeze(0)
                if core_attn_out.ndim >= 4 and core_attn_out.shape[0] == 1
                else core_attn_out,
            )
        else:
            has_mamba_track_mask = getattr(
                forward_metadata, "has_mamba_track_mask", False
            )
            use_fused_gate_cumsum = (
                _use_unifyinfer_qwen35_gdn_fused_gate_cumsum
                and not has_mamba_track_mask
                and query.shape[0] == 1
                and query_start_loc is not None
                and query_start_loc.numel() == 2
            )
            with _qwen35_gdn_trace_span("_qwen35_gdncore_gate"):
                if use_fused_gate_cumsum:
                    g, beta = fused_gdn_gating_chunk_cumsum(
                        layer.A_log, a, b, layer.dt_bias
                    )
                else:
                    g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            use_recurrent_backend_select = (
                _use_unifyinfer_qwen35_gdn_recurrent_backend_select
                and not has_mamba_track_mask
                and query.shape[0] == 1
                and query.shape[1]
                >= _unifyinfer_qwen35_gdn_recurrent_backend_select_min_tokens
                and cache_indices.numel() == 1
                and query_start_loc is not None
                and query_start_loc.numel() == 2
            )
            with _qwen35_gdn_trace_span("_qwen35_gdncore_dispatch"):
                if _use_unifyinfer_qwen35_gdn_recurrent_backend_select:
                    dispatch_extend = (
                        self.kernel_dispatcher.extend_recurrent
                        if use_recurrent_backend_select
                        else self.kernel_dispatcher.extend_chunk
                    )
                    core_attn_out, last_recurrent_state, h = dispatch_extend(
                        q=query,
                        k=key,
                        v=value,
                        g=g,
                        beta=beta,
                        ssm_states=ssm_states,
                        cache_indices=cache_indices,
                        query_start_loc=query_start_loc,
                        has_mamba_track_mask=has_mamba_track_mask,
                        g_is_chunk_cumsum=use_fused_gate_cumsum,
                    )
                else:
                    core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                        q=query,
                        k=key,
                        v=value,
                        g=g,
                        beta=beta,
                        ssm_states=ssm_states,
                        cache_indices=cache_indices,
                        query_start_loc=query_start_loc,
                        has_mamba_track_mask=has_mamba_track_mask,
                        g_is_chunk_cumsum=use_fused_gate_cumsum,
                    )

            if (is_npu() or is_cpu()) and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

        return core_attn_out
