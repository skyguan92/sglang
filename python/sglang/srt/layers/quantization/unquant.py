from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.layers.utils import MultiPlatformOp, copy_or_rebind_param
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_hip,
    is_npu,
    next_power_of_2,
    set_weight_attrs,
    use_intel_amx_backend,
    use_intel_xpu_backend,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


_is_cpu_amx_available = cpu_has_amx_support()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_disable_aiter_fused_moe = (
    get_bool_env_var("UNIFYINFER_DISABLE_AITER_FUSED_MOE") and _is_hip
)
_use_aiter_fused_moe = _use_aiter and not _disable_aiter_fused_moe
_use_qwen35_hipb_explicit_220_proj_selective = (
    get_bool_env_var("UNIFYINFER_QWEN35_HIPB_EXPLICIT_220_PROJ_SELECTIVE") and _is_hip
)
_use_qwen35_hipb_explicit_220_class_gdn_inproj = (
    get_bool_env_var("UNIFYINFER_QWEN35_HIPB_EXPLICIT_220_CLASS_GDN_INPROJ") and _is_hip
)
_use_qwen35_hipb_explicit_208_attn_qkv = (
    get_bool_env_var("UNIFYINFER_QWEN35_HIPB_EXPLICIT_208_ATTN_QKV") and _is_hip
)
_use_qwen35_hipb_explicit_192_attn_qkv = (
    get_bool_env_var("UNIFYINFER_QWEN35_HIPB_EXPLICIT_192_ATTN_QKV") and _is_hip
)
_cache_unifyinfer_prepacked_w13 = (
    get_bool_env_var("UNIFYINFER_EXPERIMENTAL_MOE_CACHE_PREPACKED_W13") and _is_hip
)
_single_storage_unifyinfer_w13 = (
    get_bool_env_var("UNIFYINFER_EXPERIMENTAL_MOE_SINGLE_STORAGE_W13") and _is_hip
)

if _use_aiter:
    from aiter.tuned_gemm import hipb_gemm, tgemm

if _use_aiter_fused_moe:
    from aiter import ActivationType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import npu_format_cast

try:
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType
except ImportError:
    flashinfer_cutlass_fused_moe = None


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["weight"])

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_intel_amx_backend(layer):
            x_shapes = x.shape
            if len(x_shapes) == 3:
                x = x.view(-1, x.shape[-1])
            output = torch.ops.sgl_kernel.weight_packed_linear(
                x,
                layer.weight,
                bias,
                True,  # is_vnni
            )
            if len(x_shapes) == 3:
                output = output.view(x_shapes[0], x_shapes[1], -1)
            return output

        elif _use_aiter and type(layer.weight.data) is torch.Tensor:
            solution_id = _maybe_get_qwen35_hipb_explicit_solution_id(x, layer.weight)
            if solution_id is not None:
                return hipb_gemm(
                    x,
                    layer.weight,
                    solution_id,
                    bias,
                    x.dtype,
                    None,
                    None,
                    None,
                    False,
                )
            return tgemm.mm(x, layer.weight, bias, otype=x.dtype)

        return F.linear(x, layer.weight, bias)


def _maybe_get_qwen35_hipb_explicit_solution_id(
    x: torch.Tensor, weight: torch.Tensor
) -> int | None:
    if x.ndim != 2:
        return None
    if x.shape[1] != 2048:
        return None
    if weight.ndim != 2:
        return None
    if weight.shape[1] != 2048:
        return None
    # `rS15ct` exhaustive sweep found that solution 5622 is the best legal
    # explicit hipBLASLt point for the two dominant fused qwen3.5 projection
    # shapes at the prompt-220 prefill point, while `4096` remained negative.
    if _use_qwen35_hipb_explicit_220_proj_selective:
        if x.shape[0] == 220 and weight.shape[0] in (12288, 9216):
            return 5622
    # `rS15cw` then showed a narrower follow-up: the `12288` GDN in-proj path
    # keeps the same winning explicit point across a bounded prompt-220-class
    # window (`192/208/220/224`), while the `9216` full-attn path only stayed
    # positive at the original `m=220` point.
    if _use_qwen35_hipb_explicit_220_class_gdn_inproj:
        if x.shape[0] in (192, 208, 220, 224) and weight.shape[0] == 12288:
            return 5622
    # `rS15cz` reopened the `9216` full-attn qkv path synthetically at
    # `m=208`, but `rS15db` kept the live contract from promoting there.
    # Keep this gate probe-only rather than part of the standing path.
    if _use_qwen35_hipb_explicit_208_attn_qkv:
        if x.shape[0] == 208 and weight.shape[0] == 9216:
            return 5622
    # `rS15cz` found an even stronger synthetic 9216 point at `m=192`, but
    # `rS15dc` was already prefill-negative on the first trusted surface.
    # Keep this gate probe-only too.
    if _use_qwen35_hipb_explicit_192_attn_qkv:
        if x.shape[0] == 192 and weight.shape[0] == 9216:
            return 5607
    return None


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):
    """MoE method without quantization."""

    def __init__(
        self, use_triton_kernels: bool = False, use_flashinfer_trtllm_moe: bool = False
    ):
        super().__init__()
        self.use_flashinfer_cutlass = get_moe_runner_backend().is_flashinfer_cutlass()
        self.use_triton_kernels = use_triton_kernels
        self.with_bias = False
        self.use_flashinfer_trtllm_moe = use_flashinfer_trtllm_moe
        self._cache_permute_indices = dict({})

    def _should_use_unifyinfer_single_storage_w13(
        self, params_dtype: torch.dtype
    ) -> bool:
        if not _single_storage_unifyinfer_w13:
            return False
        if params_dtype != torch.bfloat16:
            return False
        if (
            self.use_triton_kernels
            or self.use_flashinfer_trtllm_moe
            or self.use_flashinfer_cutlass
        ):
            return False
        try:
            if get_global_server_args().cpu_offload_gb != 0:
                return False
        except Exception:
            return False
        return True

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias

        # Fused gate_up_proj (column parallel)
        w13_up_dim = (
            2 * intermediate_size_per_partition
            if layer.moe_runner_config.is_gated
            else intermediate_size_per_partition
        )
        w13_weight_n, w13_weight_k = (w13_up_dim, hidden_size)
        if self.use_triton_kernels:
            w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n
        if self._should_use_unifyinfer_single_storage_w13(params_dtype):
            # Store one physical [E, K, N] tensor and expose the canonical
            # [E, N, K] alias to the standard Triton path via strides.
            w13_weight_storage = torch.empty(
                num_experts,
                hidden_size,
                w13_up_dim,
                dtype=params_dtype,
            )
            w13_weight_view = w13_weight_storage.as_strided(
                size=(num_experts, w13_up_dim, hidden_size),
                stride=(
                    w13_weight_storage.stride(0),
                    w13_weight_storage.stride(2),
                    w13_weight_storage.stride(1),
                ),
            )
            w13_weight = torch.nn.Parameter(
                w13_weight_view,
                requires_grad=False,
            )
            layer.unifyinfer_single_storage_w13_weight = w13_weight_storage
        else:
            w13_weight = torch.nn.Parameter(
                torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
                requires_grad=False,
            )
            layer.unifyinfer_single_storage_w13_weight = None
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        if self.with_bias:
            w13_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, w13_up_dim, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight_n, w2_weight_k = (
            hidden_size,
            intermediate_size_per_partition,
        )
        if self.use_triton_kernels:
            w2_weight_n, w2_weight_k = w2_weight_k, w2_weight_n
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.with_bias:
            w2_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Skip aiter weight shuffle when using non-auto MoE backend (e.g., triton, triton_kernels)
        # because aiter CK kernels don't support all GEMM dimensions
        _should_use_aiter_moe = (
            _use_aiter_fused_moe and get_moe_runner_backend().is_auto()
        )
        if _should_use_aiter_moe:
            copy_or_rebind_param(
                layer, "w13_weight", shuffle_weight(layer.w13_weight.data, (16, 16))
            )
            torch.cuda.empty_cache()
            copy_or_rebind_param(
                layer, "w2_weight", shuffle_weight(layer.w2_weight.data, (16, 16))
            )
            torch.cuda.empty_cache()

        # Pack weight for get better performance on CPU
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])

        # Reorder rows of W1 for fused gated activation
        if self.use_flashinfer_trtllm_moe:
            from flashinfer.fused_moe.core import (
                _maybe_get_cached_w3_w1_permute_indices,
                convert_to_block_layout,
                get_w2_permute_indices_with_cache,
            )

            # w1 and w3 have been swapped, so we don't need do that here
            epilogue_tile_m = 128
            block_k = 128
            old_shape_w13 = layer.w13_weight.data[0].shape
            old_shape_w2 = layer.w2_weight.data[0].shape
            new_shape_w13 = None
            new_shape_w2 = None
            for i in range(layer.num_local_experts):
                permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                    self._cache_permute_indices,
                    layer.w13_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights1 = (
                    layer.w13_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w13_weight.data.device)]
                    .contiguous()
                )

                permute_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    layer.w2_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights2 = (
                    layer.w2_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w2_weight.data.device)]
                    .contiguous()
                )

                tmp_weights1 = convert_to_block_layout(
                    tmp_weights1.view(torch.uint8), block_k
                )
                tmp_weights2 = convert_to_block_layout(
                    tmp_weights2.view(torch.uint8), block_k
                )

                new_shape_w13 = tmp_weights1.view(torch.bfloat16).shape
                new_shape_w2 = tmp_weights2.view(torch.bfloat16).shape
                layer.w13_weight.data[i] = (
                    tmp_weights1.view(torch.bfloat16)
                    .contiguous()
                    .reshape(old_shape_w13)
                )
                layer.w2_weight.data[i] = (
                    tmp_weights2.view(torch.bfloat16).contiguous().reshape(old_shape_w2)
                )

            layer.w13_weight.data = layer.w13_weight.data.reshape(
                layer.num_local_experts, *new_shape_w13
            )
            layer.w2_weight.data = layer.w2_weight.data.reshape(
                layer.num_local_experts, *new_shape_w2
            )

        if _is_npu:
            for weight_name in ["w13_weight", "w2_weight"]:
                weight = getattr(layer, weight_name)
                weight.data = weight.data.transpose(1, 2)
                weight.data = npu_format_cast(weight.data)

        self._maybe_cache_unifyinfer_prepacked_w13(layer)
        return

    def _maybe_cache_unifyinfer_prepacked_w13(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "unifyinfer_prepacked_w13_weight"):
            layer.unifyinfer_prepacked_w13_weight = None
        if not hasattr(layer, "unifyinfer_single_storage_w13_weight"):
            layer.unifyinfer_single_storage_w13_weight = None

        if layer.unifyinfer_single_storage_w13_weight is not None:
            layer.unifyinfer_prepacked_w13_weight = (
                layer.unifyinfer_single_storage_w13_weight
            )
            return

        if not _cache_unifyinfer_prepacked_w13:
            layer.unifyinfer_prepacked_w13_weight = None
            return

        # The standard Triton path keeps W13 in [E, N, K] for
        # invoke_fused_moe_kernel(). The mixed-tail grouped reopen wants a
        # second [E, K, N] view without paying request-time transpose cost.
        if self.use_triton_kernels or self.use_flashinfer_trtllm_moe:
            layer.unifyinfer_prepacked_w13_weight = None
            return

        w13_weight = getattr(layer, "w13_weight", None)
        if w13_weight is None or w13_weight.ndim != 3:
            layer.unifyinfer_prepacked_w13_weight = None
            return
        if w13_weight.dtype != torch.bfloat16:
            layer.unifyinfer_prepacked_w13_weight = None
            return

        try:
            layer.unifyinfer_prepacked_w13_weight = (
                w13_weight.data.transpose(1, 2).contiguous()
            )
        except torch.OutOfMemoryError as exc:
            layer.unifyinfer_prepacked_w13_weight = None
            warn = getattr(logger, "warning_once", logger.warning)
            warn(
                "Skipping experimental MoE prepacked W13 cache due to OOM: %s",
                exc,
            )

    def maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
        self,
        layer: torch.nn.Module,
        param: torch.nn.Parameter,
        weight_name: str,
    ) -> None:
        """Restore canonical BF16 MoE load shapes before hot weight copy.

        The flashinfer TRT-LLM BF16 postprocess reshapes expert weights into
        block layout. During weight update, checkpoint tensors are in
        canonical layout and need a temporary shape restore for copy.
        """
        if not get_moe_runner_backend().is_flashinfer_trtllm_routed():
            return

        expected_shape = None
        if weight_name.endswith(".experts.w13_weight"):
            w13_rows = (
                2 * layer.intermediate_size_per_partition
                if layer.moe_runner_config.is_gated
                else layer.intermediate_size_per_partition
            )
            expected_shape = (layer.num_local_experts, w13_rows, layer.hidden_size)
        elif weight_name.endswith(".experts.w2_weight"):
            expected_shape = (
                layer.num_local_experts,
                layer.hidden_size,
                layer.intermediate_size_per_partition,
            )

        if expected_shape is None or tuple(param.data.shape) == expected_shape:
            return

        expected_numel = expected_shape[0] * expected_shape[1] * expected_shape[2]
        if param.data.numel() != expected_numel:
            raise RuntimeError(
                f"Cannot restore flashinfer TRT-LLM BF16 MoE weight shape for {weight_name}: "
                f"current shape={tuple(param.data.shape)}, expected shape={expected_shape}."
            )

        param.data = param.data.reshape(expected_shape)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        if self.use_flashinfer_trtllm_moe:
            backend = (
                MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED
                if get_moe_runner_backend().is_flashinfer_trtllm_routed()
                else MoeRunnerBackend.FLASHINFER_TRTLLM
            )
        elif self.use_triton_kernels:
            backend = MoeRunnerBackend.TRITON_KERNELS
        else:
            backend = MoeRunnerBackend.TRITON
        self.runner = MoeRunner(backend, moe_runner_config)

    @property
    def load_up_proj_weight_first(self) -> bool:
        # FlashInfer CUTLASS kernel assumes [Up, Gate] Proj as W13
        return self.use_flashinfer_cutlass

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return self.forward(
            layer=layer,
            dispatch_output=dispatch_output,
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        backend = self.runner.runner_backend
        if backend.is_triton_kernels():
            from sglang.srt.layers.moe.moe_runner.triton_kernels import (
                TritonKernelsQuantInfo,
            )

            quant_info = TritonKernelsQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_bias=getattr(layer, "w13_weight_bias", None),
                w2_bias=getattr(layer, "w2_weight_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_cutlass:
            output = flashinfer_cutlass_fused_moe(
                input=x,
                token_selected_experts=topk_output.topk_ids,
                token_final_scales=topk_output.topk_weights,
                fc1_expert_weights=layer.w13_weight,
                fc2_expert_weights=layer.w2_weight,
                output_dtype=x.dtype,
                quant_scales=None,
                ep_size=layer.moe_ep_size,
                ep_rank=layer.moe_ep_rank,
                tp_size=layer.moe_tp_size,
                tp_rank=layer.moe_tp_rank,
                tune_max_num_tokens=next_power_of_2(x.shape[0]),
                activation_type=(
                    ActivationType.Relu2
                    if moe_runner_config.activation == "relu2"
                    else ActivationType.Swiglu
                ),
            )[0]
            return StandardCombineInput(hidden_states=output)
        elif self.use_flashinfer_trtllm_moe:
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                FlashInferTrtllmBf16MoeQuantInfo,
            )

            quant_info = FlashInferTrtllmBf16MoeQuantInfo(
                gemm1_weights=layer.w13_weight,
                gemm2_weights=layer.w2_weight,
                global_num_experts=layer.num_experts,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
            )
            return self.runner.run(dispatch_output, quant_info)
        else:
            # Skip aiter fused_moe when using non-auto MoE backend (e.g., triton, triton_kernels)
            # because aiter CK kernels don't support all GEMM dimensions
            _should_use_aiter_moe = (
                _use_aiter_fused_moe and get_moe_runner_backend().is_auto()
            )
            if _should_use_aiter_moe:
                assert not moe_runner_config.no_combine, "unsupported"
                topk_weights, topk_ids, _ = topk_output
                if moe_runner_config.apply_router_weight_on_input:
                    assert (
                        topk_weights.dim() == 2
                    ), "`topk_weights` should be in shape (num_tokens, topk)"
                    _, topk = topk_weights.shape
                    assert (
                        topk == 1
                    ), "Only support topk=1 when `apply_router_weight_on_input` is True"
                    x = x * topk_weights.to(x.dtype)
                    topk_weights = torch.ones_like(
                        topk_weights, dtype=torch.float32
                    )  # topk_weights must be FP32 (float32)
                try:
                    output = fused_moe(
                        x,
                        layer.w13_weight,
                        layer.w2_weight,
                        topk_weights,
                        topk_ids,
                        activation=(
                            ActivationType.Silu
                            if moe_runner_config.activation == "silu"
                            else ActivationType.Gelu
                        ),
                        expert_mask=layer.expert_mask_gpu,
                    )
                    return StandardCombineInput(hidden_states=output)
                except RuntimeError as e:
                    # AITER CK fused_moe may not support all GEMM dimensions
                    # (e.g. Gemma4 MoE with 128 experts × 704 intermediate size).
                    # Fall through to Triton MoE runner below.
                    logger.warning_once(
                        f"AITER CK fused_moe failed ({e}), "
                        "falling back to Triton MoE runner."
                    )

            quant_info = self.get_triton_quant_info(layer)
            return self.runner.run(dispatch_output, quant_info)

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        assert (
            moe_runner_config.activation == "silu"
        ), f"activation = {moe_runner_config.activation} is not supported."

        if use_intel_amx_backend(layer):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            x, topk_weights = apply_topk_weights_cpu(
                moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace # See [Note] inplace should be False in fused_experts.
                CPUQuantMethod.UNQUANT,
                None,  # w1_scale
                None,  # w2_scale
                None,  # w1_zp
                None,  # w2_zp
                None,  # block_size
                True,  # is_vnni
            )
            return StandardCombineInput(hidden_states=output)
        else:
            from sglang.srt.layers.moe.fused_moe_native import moe_forward_native

            output = moe_forward_native(
                layer,
                x,
                topk_output,
                moe_runner_config,
            )
            return StandardCombineInput(hidden_states=output)

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        w13_weight = getattr(
            layer,
            "unifyinfer_moe_w13_weight_override",
            getattr(layer, "w13_weight"),
        )
        w2_weight = getattr(
            layer,
            "unifyinfer_moe_w2_weight_override",
            getattr(layer, "w2_weight"),
        )
        w13_weight_prepacked = getattr(
            layer, "unifyinfer_moe_w13_weight_prepacked_override", None
        )
        if w13_weight_prepacked is None:
            w13_weight_prepacked = getattr(layer, "unifyinfer_prepacked_w13_weight", None)
        if w13_weight_prepacked is None:
            w13_weight_prepacked = getattr(
                layer, "unifyinfer_single_storage_w13_weight", None
            )
        if w13_weight_prepacked is None:
            if (
                isinstance(w13_weight, torch.Tensor)
                and w13_weight.ndim == 3
                and w13_weight.dtype == torch.bfloat16
                and not w13_weight.is_contiguous()
            ):
                w13_weight_prepacked = w13_weight.as_strided(
                    size=(w13_weight.shape[0], w13_weight.shape[2], w13_weight.shape[1]),
                    stride=(
                        w13_weight.stride(0),
                        w13_weight.stride(2),
                        w13_weight.stride(1),
                    ),
                )
        return TritonMoeQuantInfo(
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            w13_weight_prepacked=w13_weight_prepacked,
            b13=getattr(
                layer,
                "unifyinfer_moe_w13_weight_bias_override",
                getattr(layer, "w13_weight_bias", None),
            ),
            b2=getattr(
                layer,
                "unifyinfer_moe_w2_weight_bias_override",
                getattr(layer, "w2_weight_bias", None),
            ),
        )

    def forward_xpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config
        assert moe_runner_config.activation in [
            "silu",
            "gelu",
        ], f"activation = {moe_runner_config.activation} is not supported."

        backend = self.runner.runner_backend
        if use_intel_xpu_backend():
            # sgl-kernel-xpu path
            from sgl_kernel import fused_experts

            topk_weights, topk_ids, _ = topk_output
            output = fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                b1=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
                activation=moe_runner_config.activation,
                gemm1_alpha=moe_runner_config.gemm1_alpha,
                gemm1_limit=moe_runner_config.gemm1_clamp_limit,
            )
            return StandardCombineInput(hidden_states=output)
        else:
            assert backend.is_triton()
            assert (
                moe_runner_config.activation == "silu"
            ), f"activation = {moe_runner_config.activation} is not supported \
            for Triton PATH, please set ENV SGLANG_USE_SGL_XPU=1."

            quant_info = self.get_triton_quant_info(layer)
            return self.runner.run(dispatch_output, quant_info)

    def forward_npu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        # x.shape = [B*S, H]
        x = dispatch_output.hidden_states
        # topk_weights.shape = [B*S, K]; topk_ids.shape = [B*S, K]
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        original_dtype = x.dtype
        num_tokens = x.shape[0]
        topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.to(torch.int32)
        num_experts = layer.num_experts
        top_k = layer.top_k or topk_ids.shape[1]  # in case layer.top_k is not set

        hidden_states, expanded_row_idx, expert_tokens, _ = (
            torch.ops.npu.npu_moe_init_routing_v2(
                x,
                topk_ids,
                active_num=num_tokens * top_k,
                expert_num=num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, num_experts],
                quant_mode=-1,
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)
        w13_bias = [layer.w13_weight_bias] if self.with_bias else None
        w2_bias = [layer.w2_weight_bias] if self.with_bias else None

        # gmm1: gate_up_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            bias=w13_bias,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        # act_fn:
        if self.moe_runner_config.activation == "npu_swiglu_oai":
            from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai

            hidden_states = swiglu_oai(layer, hidden_states)
        elif self.moe_runner_config.activation == "silu":
            hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
        else:
            from sglang.srt.layers.activation import GeluAndMul

            hidden_states = GeluAndMul()(hidden_states)

        # gmm2: down_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            bias=w2_bias,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
            drop_pad_mode=2,
        )

        return StandardCombineInput(hidden_states=final_hidden_states)

    def forward_tpu(self, *args, **kwargs) -> CombineInput:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    forward_native = forward_cpu
