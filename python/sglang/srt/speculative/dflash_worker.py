import hashlib
import json
import logging
import math
import os
import time
from copy import deepcopy
from dataclasses import replace
from typing import Any, Optional, Union

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import get_last_loc
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import (
    can_dflash_use_fused_qkv_proj,
    compute_dflash_accept_len_and_bonus,
    is_dflash_sampling_verify_available,
    parse_dflash_draft_config,
    resolve_dflash_verify_mask_policy,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.utils import get_bool_env_var, is_cuda, is_cuda_alike

logger = logging.getLogger(__name__)

_FusedKVMaterializeHelper = None


def _get_fused_kv_materialize_helper():
    global _FusedKVMaterializeHelper
    if _FusedKVMaterializeHelper is None:
        from sglang.srt.speculative.triton_ops.fused_kv_materialize import (
            FusedKVMaterializeHelper,
        )

        _FusedKVMaterializeHelper = FusedKVMaterializeHelper
    return _FusedKVMaterializeHelper


class DFlashWorker:
    """DFlash speculative decoding worker (spec-v1, tp>=1/pp=1)."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.draft_window_size: Optional[int] = (
            int(server_args.speculative_dflash_draft_window_size)
            if server_args.speculative_dflash_draft_window_size is not None
            else None
        )
        self.use_compact_draft_cache = self.draft_window_size is not None
        self.device = target_worker.device

        self._warned_sampling_fallback = False
        self._logged_first_verify = False
        self._dflash_profile_enabled = os.environ.get(
            "SGLANG_DFLASH_PROFILE", ""
        ).lower() not in ("", "0", "false", "no", "off")
        self._dflash_profile_step = 0
        self._disable_fused_kv_materialize = get_bool_env_var(
            "UNIFYINFER_DFLASH_DISABLE_FUSED_KV_MATERIALIZE"
        )
        self._trace_live_layer0_kv_json = os.getenv(
            "UNIFYINFER_DFLASH_TRACE_LIVE_LAYER0_KV_JSON"
        )
        trace_live_layer0_kv_tokens = os.getenv(
            "UNIFYINFER_DFLASH_TRACE_LIVE_LAYER0_KV_TOKENS"
        )
        self._trace_live_layer0_kv_tokens = (
            int(trace_live_layer0_kv_tokens)
            if trace_live_layer0_kv_tokens is not None
            else None
        )
        self._trace_live_layer0_kv_emitted = False
        self._trace_live_draft_kv_json = os.getenv(
            "UNIFYINFER_DFLASH_TRACE_LIVE_DRAFT_KV_JSON"
        )
        trace_live_draft_kv_tokens = os.getenv(
            "UNIFYINFER_DFLASH_TRACE_LIVE_DRAFT_KV_TOKENS"
        )
        self._trace_live_draft_kv_tokens = (
            int(trace_live_draft_kv_tokens)
            if trace_live_draft_kv_tokens is not None
            else None
        )
        self._trace_live_draft_kv_emitted = False
        self._trace_live_handle_json = os.getenv(
            "UNIFYINFER_DFLASH_TRACE_LIVE_HANDLE_JSON"
        )
        trace_live_handle_tokens = os.getenv(
            "UNIFYINFER_DFLASH_TRACE_LIVE_HANDLE_TOKENS"
        )
        self._trace_live_handle_tokens = (
            int(trace_live_handle_tokens)
            if trace_live_handle_tokens is not None
            else None
        )
        self._trace_live_handle_max_events = int(
            os.getenv("UNIFYINFER_DFLASH_TRACE_LIVE_HANDLE_MAX_EVENTS", "1")
        )
        self._trace_live_handle_emitted = 0
        self._warned_true_partial_verify_unsafe_guard = False

        # Draft runner (separate KV cache + attention backend).
        # Without draft windowing, the draft worker aliases the target request->token
        # mapping and allocation state. With draft windowing enabled, the draft worker
        # keeps a private compact req->token table over the same global KV index space,
        # so radix-cache/prefix-hit KV remains reusable while draft attention sees only
        # the recent window.
        target_req_to_token_pool, target_token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        shared_req_to_token_pool = (
            None if self.use_compact_draft_cache else target_req_to_token_pool
        )
        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True
        draft_backend = draft_server_args.speculative_draft_attention_backend
        supported_draft_backends = ("flashinfer", "fa3", "fa4", "triton")
        default_draft_backend = "flashinfer" if is_cuda() else "triton"
        if draft_backend is None:
            draft_backend, _ = draft_server_args.get_attention_backends()
        if draft_backend is None:
            draft_backend = default_draft_backend
        elif draft_backend == "trtllm_mha":
            logger.warning(
                "DFLASH draft worker does not support 'trtllm_mha' because the "
                "draft path requires non-causal attention. Falling back to "
                "%r.",
                default_draft_backend,
            )
            draft_backend = default_draft_backend
        elif draft_backend not in supported_draft_backends:
            logger.warning(
                "DFLASH draft worker only supports attention_backend in %s for now, "
                "but got %r. Falling back to %r.",
                supported_draft_backends,
                draft_backend,
                default_draft_backend,
            )
            draft_backend = default_draft_backend
        # Make the draft worker backend explicit and self-contained (no further overrides).
        draft_server_args.speculative_draft_attention_backend = None
        draft_server_args.prefill_attention_backend = None
        draft_server_args.decode_attention_backend = None
        draft_server_args.attention_backend = draft_backend
        # Keep draft context length aligned with the target.
        draft_server_args.context_length = (
            target_worker.model_runner.model_config.context_len
        )
        saved_server_args = get_global_server_args()
        self.draft_worker = TpModelWorker(
            server_args=draft_server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=0,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=dp_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=shared_req_to_token_pool,
            token_to_kv_pool_allocator=target_token_to_kv_pool_allocator,
            memory_pool_config=target_worker.model_runner.memory_pool_config,
        )
        set_global_server_args_for_scheduler(saved_server_args)
        self.draft_model_runner = self.draft_worker.model_runner
        self.draft_model = self.draft_model_runner.model
        draft_config = parse_dflash_draft_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config
        )
        if server_args.speculative_num_draft_tokens is None:
            # Should not happen (ServerArgs should have inferred it), but keep a fallback.
            self.block_size = int(draft_config.resolve_block_size(default=16))
        else:
            self.block_size = int(server_args.speculative_num_draft_tokens)
            model_block_size = draft_config.block_size
            if model_block_size is None:
                model_block_size = getattr(self.draft_model, "block_size", None)
            if model_block_size is not None and int(model_block_size) != int(
                self.block_size
            ):
                logger.warning(
                    "DFLASH block size mismatch: using speculative_num_draft_tokens=%s but draft config block_size=%s.",
                    self.block_size,
                    model_block_size,
                )

        self._mask_token = draft_config.mask_token
        self._mask_token_id_override = draft_config.mask_token_id
        self._mask_token_id = self._resolve_mask_token_id(
            mask_token=self._mask_token,
            mask_token_id=self._mask_token_id_override,
        )
        if self.tp_rank == 0:
            logger.info(
                "Initialized DFLASH draft runner. attention_backend=%s, model=%s, block_size=%s, draft_window_size=%s, compact_cache=%s",
                getattr(draft_server_args, "attention_backend", None),
                self.draft_model.__class__.__name__,
                self.block_size,
                self.draft_window_size,
                self.use_compact_draft_cache,
            )
            logger.info(
                "DFLASH draft runner ready. mask_token=%s, mask_token_id=%s, mask_token_id_override=%s",
                self._mask_token,
                self._mask_token_id,
                self._mask_token_id_override,
            )

        self._block_pos_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )
        self._draft_block_ids_buf: Optional[torch.Tensor] = None  # [cap_bs, block_size]
        self._draft_block_positions_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_block_tokens_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_block_end_buf: Optional[torch.Tensor] = None  # [cap_bs]
        self._draft_seq_lens_cpu_buf: Optional[torch.Tensor] = None  # [cap_bs] on CPU
        self._draft_block_spec_info = DFlashVerifyInput(
            draft_token=torch.empty((0,), dtype=torch.long, device=self.device),
            positions=torch.empty((0,), dtype=torch.int64, device=self.device),
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._draft_greedy_gathered_max_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gathered_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gather_cap: int = 0
        self._draft_greedy_best_rank_buf: Optional[torch.Tensor] = None
        self._draft_greedy_rank_index_buf: Optional[torch.Tensor] = None
        self._draft_greedy_selected_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_index_cap: int = 0

        self._use_fused_kv_materialize = (
            is_cuda() and not self._disable_fused_kv_materialize
        )
        self._fused_kv_helper: Optional[object] = None
        if self._disable_fused_kv_materialize and self.tp_rank == 0:
            logger.info(
                "DFLASH fused KV materialization disabled by env "
                "UNIFYINFER_DFLASH_DISABLE_FUSED_KV_MATERIALIZE=1"
            )
        if self._use_fused_kv_materialize:
            self._init_fused_kv_helper()

    def _init_fused_kv_helper(self) -> None:
        """Initialize the fused KV materialization helper with pre-stacked weights."""
        try:
            layers = self.draft_model.layers
            fused_disable_reason: Optional[str] = None

            if len(layers) == 0:
                fused_disable_reason = "no layers found"

            for layer_idx, layer in enumerate(layers):
                attn = layer.self_attn
                eligible, reason = can_dflash_use_fused_qkv_proj(attn.qkv_proj)
                if not eligible:
                    fused_disable_reason = f"{reason}: layer={layer_idx}"
                    break

                # Keep semantics aligned with set_kv_buffer scaling behavior.
                k_scale = getattr(attn.attn, "k_scale", None)
                v_scale = getattr(attn.attn, "v_scale", None)
                if k_scale is not None and not math.isclose(float(k_scale), 1.0):
                    fused_disable_reason = (
                        "non-unit k_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, k_scale={k_scale}"
                    )
                    break
                if v_scale is not None and not math.isclose(float(v_scale), 1.0):
                    fused_disable_reason = (
                        "non-unit v_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, v_scale={v_scale}"
                    )
                    break

                rope_is_neox_style = bool(
                    getattr(attn.rotary_emb, "is_neox_style", True)
                )
                if not rope_is_neox_style:
                    fused_disable_reason = (
                        "non-neox RoPE is not supported for fused KV path: "
                        f"layer={layer_idx}, rope_is_neox_style={rope_is_neox_style}"
                    )
                    break

            if fused_disable_reason is not None:
                if self.tp_rank == 0:
                    logger.info(
                        "DFLASH fused KV materialization disabled: %s",
                        fused_disable_reason,
                    )
                self._use_fused_kv_materialize = False
                self._fused_kv_helper = None
                return

            FusedKVMaterializeHelper = _get_fused_kv_materialize_helper()
            first_attn = layers[0].self_attn
            rotary_emb = first_attn.rotary_emb

            self._fused_kv_helper = FusedKVMaterializeHelper(
                layers=layers,
                rotary_emb=rotary_emb,
                num_kv_heads=first_attn.num_kv_heads,
                head_dim=first_attn.head_dim,
                device=self.device,
            )
            if self.tp_rank == 0:
                logger.info(
                    "DFLASH fused KV materialization enabled. "
                    "n_layers=%d, num_kv_heads=%d, head_dim=%d",
                    len(layers),
                    first_attn.num_kv_heads,
                    first_attn.head_dim,
                )
        except Exception as e:
            logger.warning(
                "DFLASH fused KV initialization failed, falling back to sequential path: %s",
                e,
            )
            self._use_fused_kv_materialize = False
            self._fused_kv_helper = None

    def _ensure_draft_block_buffers(self, bs: int) -> None:
        cap = (
            0
            if self._draft_block_ids_buf is None
            else int(self._draft_block_ids_buf.shape[0])
        )
        if cap >= int(bs):
            return

        new_cap = max(int(bs), cap * 2 if cap > 0 else int(bs))
        device = self.device
        block_size = int(self.block_size)
        self._draft_block_ids_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_block_positions_buf = torch.empty(
            (new_cap, block_size), dtype=torch.int64, device=device
        )
        self._draft_block_tokens_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_block_end_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device=device
        )
        self._draft_seq_lens_cpu_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device="cpu"
        )

    def __getattr__(self, name):
        # Delegate anything not implemented yet to the target worker.
        return getattr(self.target_worker, name)

    def _dflash_profile_start(self):
        if not self._dflash_profile_enabled:
            return None
        if is_cuda_alike():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _dflash_profile_elapsed_ms(self, started_at) -> float:
        if started_at is None:
            return 0.0
        if is_cuda_alike():
            torch.cuda.synchronize()
        return (time.perf_counter() - started_at) * 1000.0

    def _dflash_profile_log(self, msg: str, *args) -> None:
        if self._dflash_profile_enabled and self.tp_rank == 0:
            logger.info(msg, *args)

    def clear_cache_pool(self):
        # The target worker owns the shared KV allocator/cache. For the compact
        # sliding-window path, the draft req->token view is rebuilt from committed
        # target state before each draft forward, so there is nothing persistent
        # to flush here.
        pass

    def _gather_req_to_token_masked(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        pos2d: torch.Tensor,
        mask: torch.Tensor,
        context: str,
    ) -> torch.Tensor:
        if pos2d.ndim != 2:
            raise RuntimeError(
                f"{context} expected 2D positions, got shape={tuple(pos2d.shape)}."
            )
        if mask.shape != pos2d.shape:
            raise RuntimeError(
                f"{context} mask/position shape mismatch: {tuple(mask.shape)} vs {tuple(pos2d.shape)}."
            )

        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        table_width = int(req_to_token.shape[1])
        if table_width <= 0:
            if bool(mask.any().item()):
                raise RuntimeError(
                    f"{context} req_to_token table is empty but gather mask is non-empty."
                )
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        # Only the masked-off rectangular padding can be out of range in the normal
        # ragged-batch case. Replace those don't-care columns with a valid in-range
        # position before the gather so the kernel only sees real positions.
        safe_pos2d = pos2d.masked_fill(~mask, 0)
        return req_to_token[req_pool_indices[:, None], safe_pos2d][mask].to(torch.int64)

    def _gather_req_to_token_segments(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        start: torch.Tensor | None,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        lengths = lengths.to(torch.int64)
        if lengths.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        max_len = int(lengths.max().item())
        if max_len <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)
        offsets = torch.arange(
            max_len, device=self.device, dtype=torch.int64
        ).unsqueeze(0)
        if start is None:
            pos2d = offsets.expand(req_pool_indices.shape[0], -1)
        else:
            pos2d = start.to(torch.int64).unsqueeze(1) + offsets
        mask = offsets < lengths.unsqueeze(1)
        return self._gather_req_to_token_masked(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            pos2d=pos2d,
            mask=mask,
            context="DFLASH req_to_token segment gather",
        )

    def _compute_compact_draft_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        assert self.draft_window_size is not None
        visible_lens = torch.clamp(
            seq_lens.to(dtype=torch.int32, device=self.device),
            max=int(self.draft_window_size),
        )
        if self.page_size <= 1:
            return visible_lens

        # Paged FA backends derive the page table from local token positions, so the
        # compact suffix must start on a page boundary. Keep up to page_size - 1 extra
        # tokens on the left to preserve valid local page structure.
        seq_lens_i64 = seq_lens.to(torch.int64)
        visible_lens_i64 = visible_lens.to(torch.int64)
        visible_start = seq_lens_i64 - visible_lens_i64
        aligned_start = visible_start - torch.remainder(visible_start, self.page_size)
        return (seq_lens_i64 - aligned_start).to(torch.int32)

    def _resolve_mask_token_id(
        self, *, mask_token: str, mask_token_id: Optional[int] = None
    ) -> int:
        if not isinstance(mask_token, str) or not mask_token:
            raise ValueError(
                f"DFLASH mask_token must be a non-empty string, got {mask_token!r}."
            )

        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        if mask_token_id is not None:
            resolved_id = int(mask_token_id)
            if resolved_id >= vocab_size:
                raise ValueError(
                    "DFLASH mask_token_id is outside the target vocab size. "
                    f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                    f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                    "SGLang does not support resizing target embeddings for DFLASH yet."
                )

            tokenizer = getattr(self.target_worker, "tokenizer", None)
            if tokenizer is not None:
                token_id_from_vocab = tokenizer.get_vocab().get(mask_token, None)
                if (
                    token_id_from_vocab is not None
                    and int(token_id_from_vocab) != resolved_id
                ):
                    raise ValueError(
                        "DFLASH config mismatch: dflash_config.mask_token_id conflicts with tokenizer vocab id "
                        f"for dflash_config.mask_token. mask_token={mask_token!r}, "
                        f"mask_token_id={resolved_id}, tokenizer_vocab_id={int(token_id_from_vocab)}."
                    )
            return resolved_id

        tokenizer = getattr(self.target_worker, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "DFLASH requires tokenizer initialization when dflash_config.mask_token_id is not set "
                "(skip_tokenizer_init is not supported in this mode)."
            )

        resolved_id = None
        if getattr(tokenizer, "mask_token", None) == mask_token:
            resolved_id = getattr(tokenizer, "mask_token_id", None)

        if resolved_id is None:
            # Prefer checking the explicit vocab mapping first.
            vocab = tokenizer.get_vocab()
            resolved_id = vocab.get(mask_token, None)

        if resolved_id is None:
            # Mirror the reference DFlash HF demo by adding the mask token to the tokenizer.
            # This is safe only when the resulting id stays within the target model vocab size.
            added = tokenizer.add_special_tokens({"mask_token": mask_token})
            resolved_id = getattr(tokenizer, "mask_token_id", None)
            if resolved_id is None:
                resolved_id = tokenizer.convert_tokens_to_ids(mask_token)

            if added and self.tp_rank == 0:
                logger.info(
                    "Added DFLASH mask token to tokenizer. token=%s, mask_token_id=%s, tokenizer_len=%s, model_vocab_size=%s",
                    mask_token,
                    resolved_id,
                    len(tokenizer),
                    vocab_size,
                )

        if resolved_id is None or int(resolved_id) < 0:
            raise ValueError(
                "DFLASH requires resolving a mask token id, but it could not be resolved. "
                f"mask_token={mask_token!r}."
            )

        if resolved_id >= vocab_size:
            raise ValueError(
                "DFLASH mask_token_id is outside the target vocab size. "
                f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                "SGLang does not support resizing target embeddings for DFLASH yet."
            )

        return int(resolved_id)

    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInput
    ):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return

        if batch.has_grammar:
            raise RuntimeError(
                "Invariant broken: DFLASH batch has grammar constraints, but scheduler should have rejected this request."
            )
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            if (
                not is_dflash_sampling_verify_available()
                and not self._warned_sampling_fallback
                and self.tp_rank == 0
            ):
                logger.warning(
                    "DFLASH non-greedy verification is unavailable on this build/device; "
                    "falling back to greedy argmax verification."
                )
                self._warned_sampling_fallback = True

        bs = batch.batch_size()
        self._dflash_profile_step += 1
        _profile_prepare_total = self._dflash_profile_start()
        _profile_phase = self._dflash_profile_start()

        # --- 1) Append any newly committed tokens into the draft KV cache.
        self._append_target_hidden_to_draft_kv(batch, draft_input)
        _profile_kv_prev_ms = self._dflash_profile_elapsed_ms(_profile_phase)

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if (
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DFLASH requires the target model to expose a vocab-parallel `lm_head` with `weight` and "
                "`shard_indices` attributes."
            )

        # --- 2) Draft a non-causal block with the draft model.
        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        block_ids.fill_(int(self._mask_token_id))
        block_ids[:, 0].copy_(draft_input.verified_id.to(torch.long))

        _profile_phase = self._dflash_profile_start()
        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])
        _profile_embed_ms = self._dflash_profile_elapsed_ms(_profile_phase)

        # For spec-v1, the draft KV cache is always materialized before drafting the
        # next block. `target_prefix_lens` stay absolute for RoPE; `draft_prefix_lens`
        # are the logical resident lengths in the draft-local cache.
        target_prefix_lens = batch.seq_lens  # int32, device
        draft_prefix_lens = draft_input.draft_seq_lens
        if draft_prefix_lens.dtype != torch.int32:
            draft_prefix_lens = draft_prefix_lens.to(torch.int32)
        if draft_prefix_lens.device != self.device:
            draft_prefix_lens = draft_prefix_lens.to(self.device, non_blocking=True)

        positions_2d = self._draft_block_positions_buf[:bs]
        torch.add(
            target_prefix_lens.unsqueeze(1), self._block_pos_offsets, out=positions_2d
        )
        positions = positions_2d.reshape(-1)

        block_start = draft_prefix_lens
        block_end = self._draft_block_end_buf[:bs]
        torch.add(block_start, int(self.block_size), out=block_end)

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))
        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        token_to_kv_pool_state_backup = allocator.backup_state()
        _profile_phase = self._dflash_profile_start()
        try:
            if self.page_size == 1:
                block_cache_loc = allocator.alloc(bs * self.block_size)
            else:
                block_end_cpu = seq_lens_cpu + int(self.block_size)
                last_loc = get_last_loc(
                    self.draft_model_runner.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    block_start,
                )
                block_cache_loc = allocator.alloc_extend(
                    block_start,
                    seq_lens_cpu,
                    block_end,
                    block_end_cpu,
                    last_loc,
                    bs * self.block_size,
                )
            if block_cache_loc is None:
                raise RuntimeError(
                    f"DFLASH draft OOM when allocating {bs * self.block_size} block tokens."
                )

            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                block_start,
                block_end,
                block_cache_loc,
                bs,
            )
            _profile_alloc_ms = self._dflash_profile_elapsed_ms(_profile_phase)

            # Use TARGET_VERIFY mode (cuda-graphable) to run a fixed-size draft block.
            # In this mode, `seq_lens` stores the prefix lengths; attention backends
            # derive kv_len by adding `draft_token_num`.
            draft_spec_info = self._draft_block_spec_info
            seq_lens = draft_prefix_lens
            seq_lens_sum = int(draft_prefix_lens.sum().item())
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=bs,
                input_ids=block_ids.flatten(),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=block_cache_loc,
                seq_lens_sum=seq_lens_sum,
                seq_lens_cpu=seq_lens_cpu,
                positions=positions,
                req_to_token_pool=self.draft_model_runner.req_to_token_pool,
                token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
                attn_backend=self.draft_model_runner.attn_backend,
                input_embeds=input_embeds,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                spec_info=draft_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

            _profile_phase = self._dflash_profile_start()
            with torch.inference_mode():
                draft_logits_output = self.draft_model_runner.forward(
                    forward_batch
                ).logits_output
            _profile_draft_forward_ms = self._dflash_profile_elapsed_ms(
                _profile_phase
            )
        finally:
            # Drop the speculative block from the shared allocator (EAGLE3-style).
            allocator.restore_state(token_to_kv_pool_state_backup)

        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DFLASH draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, self.block_size, -1)
        _profile_phase = self._dflash_profile_start()
        draft_next = self._greedy_sample_from_vocab_parallel_head(
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
        ).view(bs, self.block_size - 1)
        _profile_greedy_ms = self._dflash_profile_elapsed_ms(_profile_phase)
        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)
        positions = positions_2d.reshape(-1)

        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions,
            draft_token_num=self.block_size,
        )
        verify_input.profile_step = self._dflash_profile_step
        _, build_custom_mask = resolve_dflash_verify_mask_policy(
            self.model_runner.attn_backend
        )
        _profile_phase = self._dflash_profile_start()
        verify_input.prepare_for_verify(
            batch,
            self.page_size,
            build_custom_mask=build_custom_mask,
        )
        _profile_verify_prepare_ms = self._dflash_profile_elapsed_ms(_profile_phase)

        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = verify_input
        batch.return_hidden_states = False
        _profile_prepare_total_ms = self._dflash_profile_elapsed_ms(
            _profile_prepare_total
        )
        self._dflash_profile_log(
            "DFLASH profile prepare: step=%d bs=%d block_size=%d "
            "kv_prev_ms=%.3f embed_ms=%.3f alloc_ms=%.3f draft_forward_ms=%.3f "
            "greedy_lm_head_ms=%.3f verify_prepare_ms=%.3f total_ms=%.3f",
            self._dflash_profile_step,
            bs,
            self.block_size,
            _profile_kv_prev_ms,
            _profile_embed_ms,
            _profile_alloc_ms,
            _profile_draft_forward_ms,
            _profile_greedy_ms,
            _profile_verify_prepare_ms,
            _profile_prepare_total_ms,
        )

    def _greedy_sample_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Greedy argmax over the target LM head in a TP-safe way.

        We cannot materialize full logits for large vocabularies efficiently, and with
        TP>1 each rank only owns a shard of the LM head weight. This computes the
        per-rank max, gathers candidates across TP ranks, and selects the global max.
        """

        if hidden_states.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=hidden_states.device)

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        if not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):
            raise RuntimeError(
                "DFLASH greedy sampling requires a vocab-parallel head with `weight` and `shard_indices`."
            )

        shard = lm_head.shard_indices
        weight = lm_head.weight  # [local_vocab_padded, hidden]
        weight_dtype = weight.dtype

        # Valid ranges in the local shard (excluding padding):
        #   base vocab:  [0, num_org)
        #   added vocab: [num_org_padded, num_org_padded + num_added)
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        num_tokens = int(hidden_states.shape[0])
        out_token_ids = torch.empty(
            (num_tokens,), dtype=torch.long, device=hidden_states.device
        )

        def _cast_hs(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype == weight_dtype else x.to(weight_dtype)

        # Fast path (common): single-rank greedy sampling over the base vocab shard.
        # Avoids extra max/id bookkeeping that is only needed for TP sync or added vocab.
        if tp_size == 1 and num_added == 0:
            for start in range(0, num_tokens, int(chunk_size)):
                end = min(num_tokens, start + int(chunk_size))
                hs = _cast_hs(hidden_states[start:end])
                if num_org > 0:
                    base_logits = torch.matmul(hs, weight[:num_org].T)
                    out_token_ids[start:end] = (
                        torch.argmax(base_logits, dim=-1).to(torch.long)
                        + org_vocab_start
                    )
                else:
                    out_token_ids[start:end] = 0
            return out_token_ids

        for start in range(0, num_tokens, int(chunk_size)):
            end = min(num_tokens, start + int(chunk_size))
            hs = _cast_hs(hidden_states[start:end])
            chunk_len = int(hs.shape[0])

            # Base vocab logits.
            if num_org > 0:
                base_logits = torch.matmul(hs, weight[:num_org].T)
                local_max, local_arg = torch.max(base_logits, dim=-1)
            else:
                local_max = torch.full(
                    (chunk_len,),
                    torch.finfo(weight_dtype).min,
                    dtype=weight_dtype,
                    device=hs.device,
                )
                local_arg = torch.zeros(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )

            # Added vocab logits (e.g., LoRA-added embeddings), if present.
            if num_added > 0:
                added_slice_start = num_org_padded
                added_slice_end = num_org_padded + num_added
                added_logits = torch.matmul(
                    hs, weight[added_slice_start:added_slice_end].T
                )
                added_max, added_arg = torch.max(added_logits, dim=-1)
                use_added = added_max > local_max
                local_max = torch.where(use_added, added_max, local_max)
                # For base/added conversion below, keep local_arg expressed in the full local
                # weight index space (base + padding + added), matching `lm_head.weight`.
                local_arg = torch.where(
                    use_added, added_arg.to(local_arg.dtype) + num_org_padded, local_arg
                )

            # Convert local argmax indices to global token ids.
            if num_added == 0:
                local_arg.add_(org_vocab_start)
                global_ids = local_arg
            else:
                global_ids = torch.empty(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )
                is_base = local_arg < num_org
                global_ids[is_base] = org_vocab_start + local_arg[is_base]
                global_ids[~is_base] = added_vocab_start + (
                    local_arg[~is_base] - num_org_padded
                )

            if tp_size == 1:
                out_token_ids[start:end] = global_ids.to(torch.long)
                continue

            # Gather per-rank maxima and associated global ids, then select the global max.
            needed = tp_size * chunk_len
            chunk_cap = int(chunk_size)
            if (
                self._draft_greedy_gather_cap < needed
                or self._draft_greedy_gathered_max_buf is None
                or self._draft_greedy_gathered_ids_buf is None
                or self._draft_greedy_gathered_max_buf.dtype != local_max.dtype
                or self._draft_greedy_gathered_max_buf.device != hs.device
            ):
                # Allocate enough space for the max chunk size to avoid reallocations.
                cap = tp_size * chunk_cap
                self._draft_greedy_gathered_max_buf = torch.empty(
                    (cap,), dtype=local_max.dtype, device=hs.device
                )
                self._draft_greedy_gathered_ids_buf = torch.empty(
                    (cap,), dtype=global_ids.dtype, device=hs.device
                )
                self._draft_greedy_gather_cap = cap

            if (
                self._draft_greedy_index_cap < chunk_len
                or self._draft_greedy_best_rank_buf is None
                or self._draft_greedy_rank_index_buf is None
                or self._draft_greedy_selected_ids_buf is None
                or self._draft_greedy_best_rank_buf.device != hs.device
                or self._draft_greedy_selected_ids_buf.device != hs.device
            ):
                self._draft_greedy_best_rank_buf = torch.empty(
                    (chunk_cap,), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_rank_index_buf = torch.empty(
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_selected_ids_buf = torch.empty(
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_index_cap = chunk_cap

            gathered_max = self._draft_greedy_gathered_max_buf[:needed]
            gathered_ids = self._draft_greedy_gathered_ids_buf[:needed]

            tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
            tp_group.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
            gathered_max = gathered_max.view(tp_size, chunk_len)
            gathered_ids = gathered_ids.view(tp_size, chunk_len)

            best_rank = self._draft_greedy_best_rank_buf[:chunk_len]
            torch.argmax(gathered_max, dim=0, out=best_rank)

            rank_index = self._draft_greedy_rank_index_buf[:, :chunk_len]
            rank_index[0].copy_(best_rank)
            selected_ids = self._draft_greedy_selected_ids_buf[:, :chunk_len]
            torch.gather(gathered_ids, 0, rank_index, out=selected_ids)
            out_token_ids[start:end].copy_(selected_ids.view(-1))

        return out_token_ids

    def _append_target_hidden_to_draft_kv(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInput,
    ) -> None:
        """Materialize the target hidden-state features into the draft KV cache.

        This must be run before exposing new tokens to radix cache (prefix hits), otherwise
        another request could reuse target KV indices without having draft KV values.
        """

        bs = batch.batch_size()
        device = self.model_runner.device

        if draft_input.target_hidden is None:
            raise RuntimeError(
                "DFLASH draft state missing target_hidden context features."
            )
        if draft_input.ctx_lens.numel() != bs:
            raise RuntimeError(
                f"DFLASH ctx_lens length mismatch: got {draft_input.ctx_lens.numel()} for bs={bs}."
            )
        if draft_input.draft_seq_lens.numel() != bs:
            raise RuntimeError(
                f"DFLASH draft_seq_lens length mismatch: got {draft_input.draft_seq_lens.numel()} for bs={bs}."
            )

        total_ctx = int(draft_input.target_hidden.shape[0])
        if total_ctx <= 0:
            draft_input.ctx_lens = torch.zeros_like(draft_input.ctx_lens)
            draft_input.target_hidden = draft_input.target_hidden[:0]
            return

        target_req_to_token = batch.req_to_token_pool.req_to_token
        draft_req_to_token = self.draft_model_runner.req_to_token_pool.req_to_token

        req_pool_indices = batch.req_pool_indices
        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)

        ctx_lens = draft_input.ctx_lens
        if ctx_lens.dtype != torch.int32:
            ctx_lens = ctx_lens.to(torch.int32)
        if ctx_lens.device != device:
            ctx_lens = ctx_lens.to(device, non_blocking=True)
        ctx_start = batch.seq_lens.to(torch.int64) - ctx_lens.to(torch.int64)

        if bs == 1:
            # Fast path for single request.
            max_ctx = int(total_ctx)
            if max_ctx <= self._block_pos_offsets.numel():
                r = self._block_pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            pos2d = ctx_start[:, None] + r[None, :]  # [1, ctx]
            cache2d = target_req_to_token[req_pool_indices[:, None], pos2d]  # [1, ctx]
            ctx_cache_loc = cache2d.reshape(-1).to(torch.int64)  # [ctx]
            ctx_positions = pos2d.reshape(-1)  # [ctx]
        else:
            # In decode mode, ctx_lens <= block_size so we can skip the .item() sync.
            if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
                max_ctx = int(ctx_lens.max().item())
            else:
                max_ctx = int(self.block_size)
            if max_ctx <= 0:
                raise RuntimeError(f"DFLASH invalid max_ctx={max_ctx} for KV append.")

            if max_ctx <= self._block_pos_offsets.numel():
                r = self._block_pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            r = r[None, :]  # [1, max_ctx]
            pos2d = ctx_start[:, None] + r  # [bs, max_ctx]
            mask = r < ctx_lens[:, None]

            # Batched gather of cache locations and positions.
            ctx_cache_loc = self._gather_req_to_token_masked(
                req_to_token=target_req_to_token,
                req_pool_indices=req_pool_indices,
                pos2d=pos2d,
                mask=mask,
                context="DFLASH target hidden KV append",
            )  # [sum(ctx_lens)]
            ctx_positions = pos2d[mask]  # [sum(ctx_lens)]

        with torch.inference_mode():
            ctx_hidden = self.draft_model.project_target_hidden(
                draft_input.target_hidden
            )  # [sum(ctx), hidden]
            if ctx_hidden.shape[0] != ctx_cache_loc.numel():
                raise RuntimeError(
                    f"DFLASH ctx_hidden/cache_loc mismatch: {ctx_hidden.shape[0]} vs {ctx_cache_loc.numel()}."
                )
            self._maybe_trace_live_handle_metadata(
                batch=batch,
                target_hidden=draft_input.target_hidden,
                projected_hidden=ctx_hidden,
                ctx_lens=ctx_lens,
                ctx_positions=ctx_positions,
                ctx_cache_loc=ctx_cache_loc,
                req_pool_indices=req_pool_indices,
            )

            if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                try:
                    self._append_target_hidden_fused(
                        ctx_hidden, ctx_positions, ctx_cache_loc
                    )
                except Exception as e:
                    logger.warning(
                        "DFLASH fused KV append failed; falling back to sequential path: %s",
                        e,
                    )
                    self._use_fused_kv_materialize = False
                    self._fused_kv_helper = None
                    self._append_target_hidden_sequential(
                        ctx_hidden, ctx_positions, ctx_cache_loc
                    )
            else:
                self._append_target_hidden_sequential(
                    ctx_hidden, ctx_positions, ctx_cache_loc
                )

        if self.use_compact_draft_cache:
            new_draft_seq_lens = self._compute_compact_draft_seq_lens(batch.seq_lens)
            suffix_start = batch.seq_lens.to(torch.int64) - new_draft_seq_lens.to(
                torch.int64
            )
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=target_req_to_token,
                req_pool_indices=req_pool_indices,
                start=suffix_start,
                lengths=new_draft_seq_lens,
            )
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                draft_req_to_token,
                torch.zeros_like(new_draft_seq_lens),
                new_draft_seq_lens,
                suffix_cache_loc,
                bs,
            )
            draft_input.draft_seq_lens = new_draft_seq_lens
        else:
            draft_input.draft_seq_lens = batch.seq_lens.to(dtype=torch.int32)
        draft_input.ctx_lens = torch.zeros_like(ctx_lens)
        draft_input.target_hidden = draft_input.target_hidden[:0]

    def _maybe_trace_live_handle_metadata(
        self,
        *,
        batch: ScheduleBatch,
        target_hidden: torch.Tensor,
        projected_hidden: torch.Tensor,
        ctx_lens: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> None:
        if (
            self._trace_live_handle_json is None
            or self.tp_rank != 0
            or self._trace_live_handle_emitted >= self._trace_live_handle_max_events
            or (
                self._trace_live_handle_tokens is not None
                and int(projected_hidden.shape[0]) != self._trace_live_handle_tokens
            )
        ):
            return

        self._trace_live_handle_emitted += 1
        try:
            from unifyinfer.traces.sglang_target_event_export import (
                observe_dflash_target_append,
            )

            observe_dflash_target_append(
                self._trace_live_handle_json,
                batch=batch,
                target_hidden=target_hidden,
                projected_hidden=projected_hidden,
                ctx_lens=ctx_lens,
                ctx_positions=ctx_positions,
                ctx_cache_loc=ctx_cache_loc,
                token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
                req_pool_indices=req_pool_indices,
                tp_rank=int(self.tp_rank),
                block_size=int(self.block_size),
                use_compact_draft_cache=bool(self.use_compact_draft_cache),
                sync_epoch=int(self._trace_live_handle_emitted),
            )
        except Exception as e:
            logger.warning("DFLASH live handle metadata trace failed: %s", e)

    def _append_target_hidden_sequential(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        layer_observed_scalars = (
            []
            if (
                self._trace_live_draft_kv_json is not None
                and not self._trace_live_draft_kv_emitted
                and self.tp_rank == 0
            )
            else None
        )
        for layer_idx, layer in enumerate(self.draft_model.layers):
            attn = layer.self_attn
            k, v = attn.kv_proj_only(ctx_hidden)
            k = attn.apply_k_norm(k)
            self._maybe_trace_live_layer0_kv(
                layer_idx=layer_idx,
                ctx_hidden=ctx_hidden,
                ctx_positions=ctx_positions,
                k=k,
                v=v,
            )
            k = attn.apply_k_rope(ctx_positions, k)
            if layer_observed_scalars is not None:
                layer_observed_scalars.append(
                    float((k.float().sum() + v.float().sum()).item())
                )
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            self.draft_model_runner.token_to_kv_pool.set_kv_buffer(
                attn.attn,
                ctx_cache_loc,
                k,
                v,
                attn.attn.k_scale,
                attn.attn.v_scale,
            )
        self._maybe_trace_live_draft_kv(
            ctx_hidden=ctx_hidden,
            ctx_positions=ctx_positions,
            layer_observed_scalars=layer_observed_scalars,
        )

    def _maybe_trace_live_layer0_kv(
        self,
        *,
        layer_idx: int,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        if (
            self._trace_live_layer0_kv_emitted
            or self._trace_live_layer0_kv_json is None
            or self.tp_rank != 0
            or layer_idx != 0
            or (
                self._trace_live_layer0_kv_tokens is not None
                and int(ctx_hidden.shape[0]) != self._trace_live_layer0_kv_tokens
            )
        ):
            return

        self._trace_live_layer0_kv_emitted = True
        try:
            k_fp32 = k.float()
            v_fp32 = v.float()
            k_sum = float(k_fp32.sum().item())
            v_sum = float(v_fp32.sum().item())
            payload = {
                "mode": "dflash_live_layer0_kvproj_knorm",
                "layer_idx": int(layer_idx),
                "tp_rank": int(self.tp_rank),
                "ctx_hidden_shape": [int(x) for x in ctx_hidden.shape],
                "ctx_positions_shape": [int(x) for x in ctx_positions.shape],
                "ctx_positions_min": (
                    int(ctx_positions.min().item()) if ctx_positions.numel() > 0 else None
                ),
                "ctx_positions_max": (
                    int(ctx_positions.max().item()) if ctx_positions.numel() > 0 else None
                ),
                "k_shape": [int(x) for x in k.shape],
                "v_shape": [int(x) for x in v.shape],
                "k_finite": int(torch.isfinite(k).sum().item()),
                "k_nan": int(torch.isnan(k).sum().item()),
                "k_inf": int(torch.isinf(k).sum().item()),
                "v_finite": int(torch.isfinite(v).sum().item()),
                "v_nan": int(torch.isnan(v).sum().item()),
                "v_inf": int(torch.isinf(v).sum().item()),
                "k_sum": k_sum,
                "v_sum": v_sum,
                "kv_sum": float(k_sum + v_sum),
            }
            with open(self._trace_live_layer0_kv_json, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception as e:
            logger.warning("DFLASH live layer0 KV trace failed: %s", e)

    def _maybe_trace_live_draft_kv(
        self,
        *,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        layer_observed_scalars,
    ) -> None:
        if (
            self._trace_live_draft_kv_emitted
            or self._trace_live_draft_kv_json is None
            or self.tp_rank != 0
            or layer_observed_scalars is None
            or (
                self._trace_live_draft_kv_tokens is not None
                and int(ctx_hidden.shape[0]) != self._trace_live_draft_kv_tokens
            )
        ):
            return

        self._trace_live_draft_kv_emitted = True
        try:
            payload = {
                "mode": "dflash_live_draft_kv_materialize",
                "tp_rank": int(self.tp_rank),
                "ctx_hidden_shape": [int(x) for x in ctx_hidden.shape],
                "ctx_hidden_finite": int(torch.isfinite(ctx_hidden).sum().item()),
                "ctx_hidden_nan": int(torch.isnan(ctx_hidden).sum().item()),
                "ctx_hidden_inf": int(torch.isinf(ctx_hidden).sum().item()),
                "ctx_hidden_sum": float(ctx_hidden.float().sum().item()),
                "ctx_positions_shape": [int(x) for x in ctx_positions.shape],
                "ctx_positions_min": (
                    int(ctx_positions.min().item()) if ctx_positions.numel() > 0 else None
                ),
                "ctx_positions_max": (
                    int(ctx_positions.max().item()) if ctx_positions.numel() > 0 else None
                ),
                "num_draft_layers": len(layer_observed_scalars),
                "layer_observed_scalars": layer_observed_scalars,
                "total_observed_scalar": float(sum(layer_observed_scalars)),
            }
            with open(self._trace_live_draft_kv_json, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception as e:
            logger.warning("DFLASH live draft KV trace failed: %s", e)

    def _append_target_hidden_fused(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        """Fused KV materialization using batched projection + Triton kernel."""
        token_to_kv_pool = self.draft_model_runner.token_to_kv_pool
        layers = self.draft_model.layers

        def _write_layer_kv(
            layer_idx: int, cache_k: torch.Tensor, cache_v: torch.Tensor
        ) -> None:
            attn = layers[layer_idx].self_attn.attn
            token_to_kv_pool.set_kv_buffer(
                attn,
                ctx_cache_loc,
                cache_k,
                cache_v,
                attn.k_scale,
                attn.v_scale,
            )

        self._fused_kv_helper.materialize(
            ctx_hidden=ctx_hidden,
            positions=ctx_positions,
            write_layer_kv=_write_layer_kv,
        )

    def _update_target_mamba_state_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit Mamba intermediate states for accepted verify steps.

        During TARGET_VERIFY, Mamba kernels run with `disable_state_update=True` and
        cache per-step intermediate states. After acceptance, we need to commit the
        state corresponding to each request's last accepted step.
        """
        attn_backend = self.target_worker.model_runner.attn_backend
        if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            return

        accepted_steps = commit_lens.to(torch.int64) - 1
        mamba_steps_to_track = None

        if batch.mamba_track_indices is not None:
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            can_track_mask = to_track_mask & (
                to_track_ith < commit_lens.to(to_track_ith.dtype)
            )
            mamba_steps_to_track = torch.where(
                can_track_mask,
                to_track_ith.to(torch.int64),
                torch.full_like(to_track_ith, -1, dtype=torch.int64),
            )

        attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def _supports_true_partial_verify_capture(self, batch: ScheduleBatch) -> bool:
        capture_path = os.getenv("UNIFYINFER_DFLASH_TRUE_PARTIAL_VERIFY_CAPTURE_JSONL")
        if not capture_path or capture_path.lower() in ("", "0", "false", "no", "off"):
            return False
        shadow_forward = get_bool_env_var(
            "UNIFYINFER_DFLASH_TRUE_PARTIAL_VERIFY_SHADOW_FORWARD"
        )
        proof_request_forward = get_bool_env_var(
            "UNIFYINFER_DFLASH_TRUE_PARTIAL_VERIFY_PROOF_REQUEST_FORWARD"
        )
        allow_restored_shadow = get_bool_env_var(
            "UNIFYINFER_DFLASH_TRUE_PARTIAL_VERIFY_ALLOW_RESTORED_SHADOW_FORWARD"
        )
        allow_unsafe = get_bool_env_var(
            "UNIFYINFER_DFLASH_TRUE_PARTIAL_VERIFY_ALLOW_INPLACE_FORWARD"
        )
        if proof_request_forward:
            pass
        elif shadow_forward and not allow_restored_shadow:
            if not self._warned_true_partial_verify_unsafe_guard:
                logger.warning(
                    "DFLASH true partial verify restored-shadow capture requested, "
                    "but the restored same-request shadow-slot prototype is not a "
                    "safe default on ROCm. Set "
                    "UNIFYINFER_DFLASH_TRUE_PARTIAL_VERIFY_ALLOW_RESTORED_SHADOW_FORWARD=1 "
                    "only for explicit crash-repro experiments."
                )
                self._warned_true_partial_verify_unsafe_guard = True
            return False
        elif not shadow_forward and not allow_unsafe:
            if not self._warned_true_partial_verify_unsafe_guard:
                logger.warning(
                    "DFLASH true partial verify capture requested, but the current "
                    "default does not run a second target forward. Set "
                    "UNIFYINFER_DFLASH_TRUE_PARTIAL_VERIFY_PROOF_REQUEST_FORWARD=1 "
                    "for the scratch-prefix proof-request path, "
                    "UNIFYINFER_DFLASH_TRUE_PARTIAL_VERIFY_SHADOW_FORWARD=1 for "
                    "the restored shadow-slot path, or "
                    "UNIFYINFER_DFLASH_TRUE_PARTIAL_VERIFY_ALLOW_INPLACE_FORWARD=1 "
                    "only for explicit crash-repro experiments."
                )
                self._warned_true_partial_verify_unsafe_guard = True
            return False
        sampling_info = batch.sampling_info
        if sampling_info is None:
            return True
        if not getattr(sampling_info, "is_all_greedy", True):
            return False
        if getattr(sampling_info, "has_custom_logit_processor", False):
            return False
        if getattr(sampling_info, "logit_bias", None) is not None:
            return False
        penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
        if getattr(penalizer, "is_required", False):
            return False
        if getattr(batch, "has_grammar", False):
            return False
        return True

    def _true_partial_proof_request_forward_enabled(self) -> bool:
        return get_bool_env_var(
            "UNIFYINFER_DFLASH_TRUE_PARTIAL_VERIFY_PROOF_REQUEST_FORWARD"
        )

    def _true_partial_shadow_forward_enabled(self) -> bool:
        return get_bool_env_var(
            "UNIFYINFER_DFLASH_TRUE_PARTIAL_VERIFY_SHADOW_FORWARD"
        ) and get_bool_env_var(
            "UNIFYINFER_DFLASH_TRUE_PARTIAL_VERIFY_ALLOW_RESTORED_SHADOW_FORWARD"
        )

    def _true_partial_state_digests_enabled(self) -> bool:
        return get_bool_env_var(
            "UNIFYINFER_DFLASH_TRUE_PARTIAL_VERIFY_STATE_DIGESTS"
        )

    def _target_aux_hidden_capture_source(self) -> dict[str, Any]:
        model = getattr(self.target_worker.model_runner, "model", None)
        candidates = [
            model,
            getattr(model, "model", None),
            getattr(model, "language_model", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            layer_ids = getattr(candidate, "layers_to_capture", None)
            hidden_size = getattr(candidate, "hidden_size", None)
            if layer_ids is None or hidden_size is None:
                continue
            try:
                return {
                    "aux_hidden_layer_ids": [int(item) for item in layer_ids],
                    "aux_hidden_size": int(hidden_size),
                }
            except Exception:
                continue
        return {}

    def _small_tensor_list(self, tensor: Optional[torch.Tensor], *, limit: int = 32):
        if tensor is None:
            return None
        try:
            flat = tensor.detach().reshape(-1)[:limit].to(device="cpu")
            return [int(item) for item in flat.tolist()]
        except Exception as e:
            return {"error": str(e)}

    def _digest_tensor(self, tensor: torch.Tensor) -> dict[str, Any]:
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
        hasher.update(raw_bytes)
        return {
            "shape": list(cpu.shape),
            "dtype": original_dtype,
            "digest_dtype": digest_dtype,
            "digest": "sha256:" + hasher.hexdigest(),
        }

    def _digest_tensor_axis_indices(
        self,
        *,
        tensor: torch.Tensor,
        axis: int,
        indices: list[int],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        axis_i = int(axis)
        dim = int(tensor.shape[axis_i])
        for index in self._unique_ints(indices):
            if index < 0 or index >= dim:
                out.append({"index": int(index), "present": False, "dim": dim})
                continue
            selected = tensor.select(axis_i, int(index))
            out.append({"index": int(index), "present": True, **self._digest_tensor(selected)})
        return out

    def _unique_ints(self, values: list[int]) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        for value in values:
            value_i = int(value)
            if value_i in seen:
                continue
            seen.add(value_i)
            out.append(value_i)
        return out

    def _snapshot_true_partial_forward_metadata(self) -> dict[str, Any]:
        attn_backend = self.target_worker.model_runner.attn_backend
        linear_backend = getattr(attn_backend, "linear_attn_backend", attn_backend)
        metadata = getattr(linear_backend, "forward_metadata", None)
        if metadata is None:
            return {"present": False}
        return {
            "present": True,
            "class": metadata.__class__.__module__ + "." + metadata.__class__.__name__,
            "query_start_loc": self._small_tensor_list(
                getattr(metadata, "query_start_loc", None)
            ),
            "mamba_cache_indices": self._small_tensor_list(
                getattr(metadata, "mamba_cache_indices", None)
            ),
            "retrieve_next_token": self._small_tensor_list(
                getattr(metadata, "retrieve_next_token", None)
            ),
            "retrieve_next_sibling": self._small_tensor_list(
                getattr(metadata, "retrieve_next_sibling", None)
            ),
            "retrieve_parent_token": self._small_tensor_list(
                getattr(metadata, "retrieve_parent_token", None)
            ),
        }

    def _snapshot_true_partial_recurrent_state(
        self,
        *,
        batch: ScheduleBatch,
        stage: str,
        source_req_pool_indices: list[int],
        proof_req_pool_indices: list[int],
        partial_width: int,
    ) -> dict[str, Any] | None:
        if not self._true_partial_state_digests_enabled():
            return None
        try:
            req_to_token_pool = batch.req_to_token_pool
            req_device = batch.req_pool_indices.device
            source_req_tensor = torch.tensor(
                source_req_pool_indices,
                dtype=batch.req_pool_indices.dtype,
                device=req_device,
            )
            proof_req_tensor = torch.tensor(
                proof_req_pool_indices,
                dtype=batch.req_pool_indices.dtype,
                device=req_device,
            )
            source_mamba_indices = (
                req_to_token_pool.get_mamba_indices(source_req_tensor)
                if source_req_pool_indices
                else torch.empty((0,), dtype=torch.int32, device=req_device)
            )
            proof_mamba_indices = (
                req_to_token_pool.get_mamba_indices(proof_req_tensor)
                if proof_req_pool_indices
                else torch.empty((0,), dtype=torch.int32, device=req_device)
            )
            source_mamba_cpu = [
                int(item)
                for item in source_mamba_indices.detach().to(device="cpu").tolist()
            ]
            proof_mamba_cpu = [
                int(item)
                for item in proof_mamba_indices.detach().to(device="cpu").tolist()
            ]
            caches = req_to_token_pool.get_speculative_mamba2_params_all_layers()
            req_indices = self._unique_ints(source_req_pool_indices + proof_req_pool_indices)
            mamba_indices = self._unique_ints(source_mamba_cpu + proof_mamba_cpu)
            snapshot = {
                "stage": stage,
                "partial_width": int(partial_width),
                "source_req_pool_indices": [int(item) for item in source_req_pool_indices],
                "proof_req_pool_indices": [int(item) for item in proof_req_pool_indices],
                "source_mamba_indices": source_mamba_cpu,
                "proof_mamba_indices": proof_mamba_cpu,
                "forward_metadata": self._snapshot_true_partial_forward_metadata(),
                "state_digests": {
                    "temporal_by_mamba_index": self._digest_tensor_axis_indices(
                        tensor=caches.temporal,
                        axis=1,
                        indices=mamba_indices,
                    ),
                    "intermediate_ssm_by_req_index": self._digest_tensor_axis_indices(
                        tensor=caches.intermediate_ssm,
                        axis=1,
                        indices=req_indices,
                    ),
                },
            }
            conv_cache = getattr(caches, "conv", None)
            if conv_cache is not None and len(conv_cache) > 0:
                snapshot["state_digests"]["conv0_by_mamba_index"] = (
                    self._digest_tensor_axis_indices(
                        tensor=conv_cache[0],
                        axis=1,
                        indices=mamba_indices,
                    )
                )
            intermediate_conv_cache = getattr(caches, "intermediate_conv_window", None)
            if intermediate_conv_cache is not None and len(intermediate_conv_cache) > 0:
                snapshot["state_digests"]["intermediate_conv0_by_req_index"] = (
                    self._digest_tensor_axis_indices(
                        tensor=intermediate_conv_cache[0],
                        axis=1,
                        indices=req_indices,
                    )
                )
            return snapshot
        except Exception as e:
            return {
                "stage": stage,
                "error": str(e),
            }

    def _build_true_partial_verify_mask(
        self,
        *,
        batch: ScheduleBatch,
        partial_width: int,
    ) -> torch.Tensor:
        mask_chunks = []
        q_idx = torch.arange(
            partial_width, device=batch.device, dtype=torch.int32
        ).unsqueeze(1)
        for prefix_len in batch.seq_lens_cpu.tolist():
            prefix_len_i = int(prefix_len)
            kv_len = prefix_len_i + partial_width
            k_idx = torch.arange(
                kv_len, device=batch.device, dtype=torch.int32
            ).unsqueeze(0)
            mask_chunks.append((k_idx <= (prefix_len_i + q_idx)).flatten())
        return (
            torch.cat(mask_chunks, dim=0)
            if mask_chunks
            else torch.empty((0,), dtype=torch.bool, device=batch.device)
        )

    def _maybe_capture_true_partial_verify_forward(
        self,
        *,
        batch: ScheduleBatch,
        model_worker_batch,
        verify_input: DFlashVerifyInput,
        logits_output,
        forward_kwargs: dict,
        pre_verify_recurrent_seed: dict[str, Any] | None = None,
    ) -> None:
        if not self._supports_true_partial_verify_capture(batch):
            return
        try:
            hidden = logits_output.hidden_states
            if hidden is None:
                return
            bs = batch.batch_size()
            if bs != 1:
                logger.warning(
                    "DFLASH true partial verify capture currently supports bs=1, got bs=%s.",
                    bs,
                )
                return

            full_width = int(verify_input.draft_token_num)
            candidates = verify_input.draft_token.view(bs, full_width)
            full_target_predict = torch.argmax(
                logits_output.next_token_logits, dim=-1
            ).view(bs, full_width)
            accept_len, _ = compute_dflash_accept_len_and_bonus(
                candidates=candidates,
                target_predict=full_target_predict,
            )
            prefix_lens = (accept_len.to(torch.int64) + 1).clamp(
                min=1,
                max=full_width,
            )
            partial_width = int(prefix_lens.max().item())
            if partial_width >= full_width:
                # This row would be a full-width rerun, not a partial-forward proof.
                return

            positions = verify_input.positions.view(bs, full_width)
            full_positions_cpu = positions.detach().to(device="cpu", dtype=torch.int64)
            partial_positions_cpu = full_positions_cpu[:, :partial_width].contiguous()
            partial_spec = DFlashVerifyInput(
                draft_token=candidates[:, :partial_width].reshape(-1).contiguous(),
                positions=positions[:, :partial_width].reshape(-1).contiguous(),
                draft_token_num=partial_width,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                num_tokens_per_batch=partial_width,
            )
            _, build_custom_mask = resolve_dflash_verify_mask_policy(
                self.model_runner.attn_backend
            )
            if build_custom_mask:
                partial_spec.custom_mask = self._build_true_partial_verify_mask(
                    batch=batch,
                    partial_width=partial_width,
                )

            if self._true_partial_proof_request_forward_enabled():
                partial_result, shadow_plan = (
                    self._run_true_partial_verify_proof_request_forward(
                        batch=batch,
                        model_worker_batch=model_worker_batch,
                        partial_spec=partial_spec,
                        partial_width=partial_width,
                        forward_kwargs=forward_kwargs,
                        pre_verify_recurrent_seed=pre_verify_recurrent_seed,
                    )
                )
                if partial_result is None:
                    return
            elif self._true_partial_shadow_forward_enabled():
                partial_result, shadow_plan = (
                    self._run_true_partial_verify_shadow_forward(
                        batch=batch,
                        model_worker_batch=model_worker_batch,
                        partial_spec=partial_spec,
                        partial_width=partial_width,
                        forward_kwargs=forward_kwargs,
                    )
                )
                if partial_result is None:
                    return
            else:
                out_cache_loc = model_worker_batch.out_cache_loc.view(bs, full_width)
                shadow_plan = {
                    "mode": "unsafe_inplace_reused_full_verify_slots",
                    "eligible": True,
                }
                partial_worker_batch = replace(
                    model_worker_batch,
                    input_ids=partial_spec.draft_token,
                    out_cache_loc=out_cache_loc[:, :partial_width]
                    .reshape(-1)
                    .contiguous(),
                    spec_info=partial_spec,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                )
                partial_result = self.target_worker.forward_batch_generation(
                    partial_worker_batch,
                    is_verify=True,
                    **forward_kwargs,
                )
            partial_logits = partial_result.logits_output
            partial_hidden = partial_logits.hidden_states
            if partial_hidden is None:
                return
            partial_target_predict = torch.argmax(
                partial_logits.next_token_logits, dim=-1
            ).view(bs, partial_width)
            full_boundary_digests = getattr(
                logits_output,
                "unifyinfer_qwen35_dflash_boundary_digests",
                None,
            )
            partial_boundary_digests = getattr(
                partial_logits,
                "unifyinfer_qwen35_dflash_boundary_digests",
                None,
            )

            from unifyinfer.traces.dflash_partial_verify_capture import (
                maybe_append_dflash_true_partial_verify_capture,
            )

            maybe_append_dflash_true_partial_verify_capture(
                candidates=candidates,
                full_target_predict=full_target_predict,
                full_hidden=hidden.view(bs, full_width, -1),
                partial_target_predict=partial_target_predict,
                partial_hidden=partial_hidden.view(bs, partial_width, -1),
                prefix_lens=prefix_lens.detach().cpu().tolist(),
                profile_step=getattr(verify_input, "profile_step", None),
                req_ids=[
                    getattr(req, "rid", getattr(req, "request_id", index))
                    for index, req in enumerate(batch.reqs)
                ],
                source={
                    "hook": "DFlashWorker._maybe_capture_true_partial_verify_forward",
                    "trace_env": "UNIFYINFER_DFLASH_TRUE_PARTIAL_VERIFY_CAPTURE_JSONL",
                    "forward_mode": str(batch.forward_mode),
                    "seq_lens_before_verify": [
                        int(item) for item in batch.seq_lens_cpu.tolist()
                    ],
                    "full_positions": full_positions_cpu.tolist(),
                    "partial_positions": partial_positions_cpu.tolist(),
                    "partial_positions_match_full_prefix": bool(
                        torch.equal(full_positions_cpu[:, :partial_width], partial_positions_cpu)
                    ),
                    "req_pool_indices": [
                        int(item) for item in batch.req_pool_indices.detach().cpu().tolist()
                    ],
                    "out_cache_loc_shape": list(model_worker_batch.out_cache_loc.shape),
                    **self._target_aux_hidden_capture_source(),
                    "boundary_digest_probe": {
                        "enabled": bool(
                            full_boundary_digests is not None
                            or partial_boundary_digests is not None
                        ),
                        "env": "UNIFYINFER_QWEN35_DFLASH_BOUNDARY_DIGESTS",
                        "full": full_boundary_digests or [],
                        "partial": partial_boundary_digests or [],
                    },
                    "shadow_forward_plan": shadow_plan,
                },
            )
        except Exception as e:
            logger.warning("DFLASH true partial verify capture failed: %s", e)

    def _alloc_true_partial_proof_tail_slots(
        self,
        *,
        allocator,
        bs: int,
        partial_width: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        allocation_width = int(partial_width)
        if self.page_size > 1:
            allocation_width = (
                (allocation_width + int(self.page_size) - 1) // int(self.page_size)
            ) * int(self.page_size)
        allocated = allocator.alloc(int(bs) * allocation_width)
        if allocated is None:
            return None, None, allocation_width
        if allocation_width == int(partial_width):
            return allocated, allocated, allocation_width
        visible = (
            allocated.view(int(bs), allocation_width)[:, : int(partial_width)]
            .reshape(-1)
            .contiguous()
        )
        return visible, allocated, allocation_width

    def _linear_forward_metadata_holder(self):
        attn_backend = self.target_worker.model_runner.attn_backend
        return getattr(attn_backend, "linear_attn_backend", attn_backend)

    def _backup_true_partial_intermediate_cache(
        self,
        *,
        batch: ScheduleBatch,
        bs: int,
    ) -> dict[str, Any] | None:
        req_to_token_pool = batch.req_to_token_pool
        if not hasattr(req_to_token_pool, "get_speculative_mamba2_params_all_layers"):
            return None
        try:
            caches = req_to_token_pool.get_speculative_mamba2_params_all_layers()
            if not hasattr(caches, "intermediate_ssm"):
                return None
            indices = torch.arange(
                int(bs),
                dtype=torch.long,
                device=caches.intermediate_ssm.device,
            )
            backup: dict[str, Any] = {
                "indices": indices,
                "intermediate_ssm": caches.intermediate_ssm[:, indices]
                .detach()
                .clone(),
                "intermediate_conv_window": [],
            }
            intermediate_conv_cache = getattr(caches, "intermediate_conv_window", None)
            if intermediate_conv_cache is not None:
                backup["intermediate_conv_window"] = [
                    (idx, tensor[:, indices].detach().clone())
                    for idx, tensor in enumerate(intermediate_conv_cache)
                ]
            return backup
        except Exception as e:
            logger.warning(
                "DFLASH true partial proof-request intermediate-cache backup failed: %s",
                e,
            )
            return None

    def _restore_true_partial_intermediate_cache(
        self,
        *,
        batch: ScheduleBatch,
        backup: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if backup is None:
            return {"present": False, "restored": False}
        try:
            caches = batch.req_to_token_pool.get_speculative_mamba2_params_all_layers()
            indices = backup["indices"]
            caches.intermediate_ssm[:, indices] = backup["intermediate_ssm"]
            restored_conv = 0
            intermediate_conv_cache = getattr(caches, "intermediate_conv_window", None)
            if intermediate_conv_cache is not None:
                for idx, saved in backup.get("intermediate_conv_window", []):
                    intermediate_conv_cache[idx][:, indices] = saved
                    restored_conv += 1
            return {
                "present": True,
                "restored": True,
                "indices": [int(item) for item in indices.detach().cpu().tolist()],
                "conv_windows_restored": restored_conv,
            }
        except Exception as e:
            logger.warning(
                "DFLASH true partial proof-request intermediate-cache restore failed: %s",
                e,
            )
            return {"present": True, "restored": False, "error": str(e)}

    def _backup_true_partial_recurrent_seed(
        self,
        *,
        batch: ScheduleBatch,
        bs: int,
    ) -> dict[str, Any] | None:
        req_to_token_pool = batch.req_to_token_pool
        mamba_pool = getattr(req_to_token_pool, "mamba_pool", None)
        if (
            mamba_pool is None
            or not hasattr(req_to_token_pool, "get_speculative_mamba2_params_all_layers")
            or not hasattr(req_to_token_pool, "get_mamba_indices")
        ):
            return None
        try:
            source_mamba_indices = req_to_token_pool.get_mamba_indices(
                batch.req_pool_indices
            ).to(dtype=torch.long)
            caches = req_to_token_pool.get_speculative_mamba2_params_all_layers()
            intermediate_indices = torch.arange(
                int(bs),
                dtype=torch.long,
                device=caches.intermediate_ssm.device,
            )
            backup: dict[str, Any] = {
                "present": True,
                "source_mamba_indices": source_mamba_indices.detach().clone(),
                "intermediate_indices": intermediate_indices,
                "conv": [
                    (idx, tensor[:, source_mamba_indices].detach().clone())
                    for idx, tensor in enumerate(getattr(caches, "conv", []))
                ],
                "temporal": caches.temporal[:, source_mamba_indices]
                .detach()
                .clone(),
                "intermediate_ssm": caches.intermediate_ssm[:, intermediate_indices]
                .detach()
                .clone(),
                "intermediate_conv_window": [],
            }
            intermediate_conv_cache = getattr(caches, "intermediate_conv_window", None)
            if intermediate_conv_cache is not None:
                backup["intermediate_conv_window"] = [
                    (idx, tensor[:, intermediate_indices].detach().clone())
                    for idx, tensor in enumerate(intermediate_conv_cache)
                ]
            return backup
        except Exception as e:
            logger.warning(
                "DFLASH true partial proof-request recurrent seed backup failed: %s",
                e,
            )
            return {"present": True, "error": str(e)}

    def _seed_true_partial_recurrent_state(
        self,
        *,
        batch: ScheduleBatch,
        backup: dict[str, Any] | None,
        proof_mamba_indices: torch.Tensor | None,
    ) -> dict[str, Any]:
        if not backup or backup.get("error"):
            return {"present": bool(backup), "seeded": False, "error": backup.get("error") if backup else None}
        if proof_mamba_indices is None:
            return {"present": True, "seeded": False, "error": "proof_mamba_indices_missing"}
        try:
            caches = batch.req_to_token_pool.get_speculative_mamba2_params_all_layers()
            proof_indices = proof_mamba_indices.to(
                device=caches.temporal.device,
                dtype=torch.long,
            )
            for idx, saved in backup.get("conv", []):
                caches.conv[int(idx)][:, proof_indices] = saved.to(
                    device=caches.conv[int(idx)].device,
                    dtype=caches.conv[int(idx)].dtype,
                )
            caches.temporal[:, proof_indices] = backup["temporal"].to(
                device=caches.temporal.device,
                dtype=caches.temporal.dtype,
            )

            intermediate_indices = backup["intermediate_indices"].to(
                device=caches.intermediate_ssm.device,
                dtype=torch.long,
            )
            caches.intermediate_ssm[:, intermediate_indices] = backup[
                "intermediate_ssm"
            ].to(
                device=caches.intermediate_ssm.device,
                dtype=caches.intermediate_ssm.dtype,
            )
            intermediate_conv_cache = getattr(caches, "intermediate_conv_window", None)
            if intermediate_conv_cache is not None:
                for idx, saved in backup.get("intermediate_conv_window", []):
                    intermediate_conv_cache[int(idx)][:, intermediate_indices] = saved.to(
                        device=intermediate_conv_cache[int(idx)].device,
                        dtype=intermediate_conv_cache[int(idx)].dtype,
                    )
            return {
                "present": True,
                "seeded": True,
                "proof_mamba_indices": [
                    int(item) for item in proof_indices.detach().cpu().tolist()
                ],
                "intermediate_indices": [
                    int(item) for item in intermediate_indices.detach().cpu().tolist()
                ],
            }
        except Exception as e:
            logger.warning(
                "DFLASH true partial proof-request recurrent seed restore failed: %s",
                e,
            )
            return {"present": True, "seeded": False, "error": str(e)}

    def _run_true_partial_verify_proof_request_forward(
        self,
        *,
        batch: ScheduleBatch,
        model_worker_batch: ModelWorkerBatch,
        partial_spec: DFlashVerifyInput,
        partial_width: int,
        forward_kwargs: dict,
        pre_verify_recurrent_seed: dict[str, Any] | None = None,
    ):
        """Run the proof forward under a temporary scratch req-pool identity."""

        from unifyinfer.traces.dflash_true_partial_shadow import (
            build_dflash_true_partial_proof_request_forward_plan,
        )

        bs = batch.batch_size()
        req_to_token_pool = batch.req_to_token_pool
        req_to_token = req_to_token_pool.req_to_token
        prefix_lens_cpu = [int(item) for item in batch.seq_lens_cpu.tolist()]
        free_slots = list(req_to_token_pool.free_slots)
        proof_req_pool_indices_cpu = [int(item) for item in free_slots[:bs]]
        has_mamba_mapping = hasattr(
            req_to_token_pool, "req_index_to_mamba_index_mapping"
        )
        proof_plan = build_dflash_true_partial_proof_request_forward_plan(
            prefix_lens=prefix_lens_cpu,
            full_width=int(getattr(model_worker_batch.spec_info, "draft_token_num", 0)),
            partial_width=int(partial_width),
            page_size=int(self.page_size),
            req_pool_available=len(free_slots),
            req_to_token_table_width=int(req_to_token.shape[1]),
            mamba_pool_available=(
                batch.req_to_token_pool.mamba_pool.available_size()
                if has_mamba_mapping and hasattr(batch.req_to_token_pool, "mamba_pool")
                else None
            ),
            mamba_ping_pong_track_buffer_width=(
                int(
                    getattr(
                        batch.req_to_token_pool,
                        "mamba_ping_pong_track_buffer_size",
                        0,
                    )
                )
                if hasattr(
                    batch.req_to_token_pool,
                    "req_index_to_mamba_ping_pong_track_buffer_mapping",
                )
                else 0
            ),
            allocator_class=batch.token_to_kv_pool_allocator.__class__.__module__
            + "."
            + batch.token_to_kv_pool_allocator.__class__.__name__,
            req_to_token_pool_class=req_to_token_pool.__class__.__module__
            + "."
            + req_to_token_pool.__class__.__name__,
            has_mamba_mapping=has_mamba_mapping,
        )
        if not proof_plan["eligible"]:
            logger.warning(
                "DFLASH true partial proof-request forward rejected: %s",
                proof_plan.get("rejection_reason"),
            )
            return None, proof_plan

        allocator = batch.token_to_kv_pool_allocator
        allocator_state = allocator.backup_state()
        saved_free_slots = list(req_to_token_pool.free_slots)
        saved_req_rows: list[tuple[int, int, torch.Tensor]] = []
        saved_mamba_rows: list[tuple[torch.Tensor, int, torch.Tensor]] = []
        mamba_mapping_copies: list[dict] = []
        mamba_state_forks: list[dict] = []
        mamba_pool = getattr(req_to_token_pool, "mamba_pool", None)
        saved_mamba_free_slots = (
            mamba_pool.free_slots.detach().clone() if mamba_pool is not None else None
        )
        intermediate_cache_backup = self._backup_true_partial_intermediate_cache(
            batch=batch,
            bs=bs,
        )
        restore_report: dict[str, Any] = {}
        metadata_holder = self._linear_forward_metadata_holder()
        saved_forward_metadata = getattr(metadata_holder, "forward_metadata", None)
        seed_report: dict[str, Any] = {"present": False, "seeded": False}
        proof_req_pool_indices = torch.tensor(
            proof_req_pool_indices_cpu,
            dtype=batch.req_pool_indices.dtype,
            device=batch.req_pool_indices.device,
        )
        source_req_pool_indices = [
            int(item) for item in batch.req_pool_indices.detach().cpu().tolist()
        ]
        recurrent_state_probe: dict[str, Any] | None = (
            {"schema_version": 1, "artifact": "dflash_true_partial_recurrent_state_probe", "snapshots": []}
            if self._true_partial_state_digests_enabled()
            else None
        )

        def _append_recurrent_snapshot(stage: str) -> None:
            if recurrent_state_probe is None:
                return
            snapshot = self._snapshot_true_partial_recurrent_state(
                batch=batch,
                stage=stage,
                source_req_pool_indices=source_req_pool_indices,
                proof_req_pool_indices=proof_req_pool_indices_cpu,
                partial_width=partial_width,
            )
            if snapshot is not None:
                recurrent_state_probe["snapshots"].append(snapshot)

        try:
            _append_recurrent_snapshot("before_proof_request_setup")
            req_to_token_pool.free_slots = req_to_token_pool.free_slots[bs:]
            proof_cache_loc, allocated_cache_loc, allocation_width = (
                self._alloc_true_partial_proof_tail_slots(
                    allocator=allocator,
                    bs=bs,
                    partial_width=partial_width,
                )
            )
            if proof_cache_loc is None or allocated_cache_loc is None:
                proof_plan = {
                    **proof_plan,
                    "eligible": False,
                    "rejection_reason": "proof_tail_kv_slot_allocation_failed",
                }
                logger.warning("DFLASH true partial proof-request forward rejected: OOM")
                return None, proof_plan

            proof_cache_loc_2d = proof_cache_loc.view(bs, int(partial_width))
            proof_mamba_indices = None
            if has_mamba_mapping:
                if mamba_pool is None:
                    proof_plan = {
                        **proof_plan,
                        "eligible": False,
                        "rejection_reason": "proof_mamba_pool_unavailable",
                    }
                    logger.warning(
                        "DFLASH true partial proof-request forward rejected: no mamba pool"
                    )
                    return None, proof_plan
                source_mamba_indices = req_to_token_pool.get_mamba_indices(
                    batch.req_pool_indices
                ).to(dtype=torch.long, device=batch.req_pool_indices.device)
                proof_mamba_indices = mamba_pool.alloc(bs)
                if proof_mamba_indices is None:
                    proof_plan = {
                        **proof_plan,
                        "eligible": False,
                        "rejection_reason": "proof_mamba_slot_allocation_failed",
                    }
                    logger.warning(
                        "DFLASH true partial proof-request forward rejected: mamba OOM"
                    )
                    return None, proof_plan
                proof_mamba_indices = proof_mamba_indices.to(
                    dtype=torch.long,
                    device=source_mamba_indices.device,
                )
                mamba_pool.copy_from(source_mamba_indices, proof_mamba_indices)
                mapping = req_to_token_pool.req_index_to_mamba_index_mapping
                for row_index, proof_req_pool_index in enumerate(proof_req_pool_indices_cpu):
                    proof_req_pool_index_i = int(proof_req_pool_index)
                    saved_mamba_rows.append(
                        (
                            mapping,
                            proof_req_pool_index_i,
                            mapping[proof_req_pool_index_i].detach().clone(),
                        )
                    )
                    mapping[proof_req_pool_index_i] = proof_mamba_indices[
                        row_index
                    ].to(dtype=mapping.dtype)
                    source_idx_i = int(source_mamba_indices[row_index].detach().cpu())
                    proof_idx_i = int(proof_mamba_indices[row_index].detach().cpu())
                    mamba_state_forks.append(
                        {
                            "attr": "req_index_to_mamba_index_mapping",
                            "source_req_pool_index": source_req_pool_indices[row_index],
                            "proof_req_pool_index": proof_req_pool_index_i,
                            "source_mamba_index": source_idx_i,
                            "proof_mamba_index": proof_idx_i,
                            "distinct": source_idx_i != proof_idx_i,
                            "source_copied_to_proof": True,
                        }
                    )

                track_mapping = getattr(
                    req_to_token_pool,
                    "req_index_to_mamba_ping_pong_track_buffer_mapping",
                    None,
                )
                if track_mapping is not None:
                    source_track = track_mapping[batch.req_pool_indices].to(
                        dtype=torch.long,
                        device=batch.req_pool_indices.device,
                    )
                    track_flat = source_track.reshape(-1)
                    proof_track_flat = mamba_pool.alloc(int(track_flat.numel()))
                    if proof_track_flat is None:
                        proof_plan = {
                            **proof_plan,
                            "eligible": False,
                            "rejection_reason": (
                                "proof_mamba_ping_pong_slot_allocation_failed"
                            ),
                        }
                        logger.warning(
                            "DFLASH true partial proof-request forward rejected: "
                            "mamba ping-pong OOM"
                        )
                        return None, proof_plan
                    proof_track_flat = proof_track_flat.to(
                        dtype=torch.long,
                        device=track_flat.device,
                    )
                    mamba_pool.copy_from(track_flat, proof_track_flat)
                    proof_track = proof_track_flat.view_as(source_track)
                    for row_index, proof_req_pool_index in enumerate(
                        proof_req_pool_indices_cpu
                    ):
                        proof_req_pool_index_i = int(proof_req_pool_index)
                        saved_mamba_rows.append(
                            (
                                track_mapping,
                                proof_req_pool_index_i,
                                track_mapping[proof_req_pool_index_i]
                                .detach()
                                .clone(),
                            )
                        )
                        track_mapping[proof_req_pool_index_i] = proof_track[
                            row_index
                        ].to(dtype=track_mapping.dtype)
                        source_values = [
                            int(item)
                            for item in source_track[row_index]
                            .detach()
                            .cpu()
                            .reshape(-1)
                            .tolist()
                        ]
                        proof_values = [
                            int(item)
                            for item in proof_track[row_index]
                            .detach()
                            .cpu()
                            .reshape(-1)
                            .tolist()
                        ]
                        mamba_state_forks.append(
                            {
                                "attr": (
                                    "req_index_to_mamba_ping_pong_track_buffer_mapping"
                                ),
                                "source_req_pool_index": source_req_pool_indices[
                                    row_index
                                ],
                                "proof_req_pool_index": proof_req_pool_index_i,
                                "source_mamba_indices": source_values,
                                "proof_mamba_indices": proof_values,
                                "distinct": not (
                                    set(source_values) & set(proof_values)
                                ),
                                "source_copied_to_proof": True,
                            }
                        )

            max_restore_end = max(
                int(prefix_len) + int(partial_width)
                for prefix_len in prefix_lens_cpu
            )
            for proof_req_pool_index in proof_req_pool_indices_cpu:
                saved_req_rows.append(
                    (
                        int(proof_req_pool_index),
                        max_restore_end,
                        req_to_token[
                            int(proof_req_pool_index), :max_restore_end
                        ].detach().clone(),
                    )
                )

            for row_index, (source_req_pool_index, proof_req_pool_index, prefix_len) in enumerate(
                zip(
                    source_req_pool_indices,
                    proof_req_pool_indices_cpu,
                    prefix_lens_cpu,
                    strict=True,
                )
            ):
                source_req_pool_index_i = int(source_req_pool_index)
                proof_req_pool_index_i = int(proof_req_pool_index)
                prefix_len_i = int(prefix_len)
                end = prefix_len_i + int(partial_width)
                req_to_token[proof_req_pool_index_i, :prefix_len_i] = req_to_token[
                    source_req_pool_index_i, :prefix_len_i
                ]
                req_to_token[proof_req_pool_index_i, prefix_len_i:end] = (
                    proof_cache_loc_2d[row_index].to(req_to_token.dtype)
                )

            seed_report = self._seed_true_partial_recurrent_state(
                batch=batch,
                backup=pre_verify_recurrent_seed,
                proof_mamba_indices=proof_mamba_indices,
            )
            _append_recurrent_snapshot("before_partial_forward")
            partial_worker_batch = replace(
                model_worker_batch,
                input_ids=partial_spec.draft_token,
                req_pool_indices=proof_req_pool_indices,
                out_cache_loc=proof_cache_loc,
                spec_info=partial_spec,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            partial_result = self.target_worker.forward_batch_generation(
                partial_worker_batch,
                is_verify=True,
                **forward_kwargs,
            )
            if is_cuda_alike():
                torch.cuda.synchronize()
            _append_recurrent_snapshot("after_partial_forward")
            return partial_result, {
                **proof_plan,
                "source_req_pool_indices": source_req_pool_indices,
                "proof_req_pool_indices": proof_req_pool_indices_cpu,
                "allocated_tail_slots": int(proof_cache_loc.numel()),
                "allocated_tail_allocation_slots": int(allocated_cache_loc.numel()),
                "tail_allocation_width_per_request": int(allocation_width),
                "mamba_mapping_copies": mamba_mapping_copies,
                "mamba_state_forks": mamba_state_forks,
                "pre_verify_recurrent_seed": seed_report,
                "restore_report": restore_report,
                **(
                    {"recurrent_state_probe": recurrent_state_probe}
                    if recurrent_state_probe is not None
                    else {}
                ),
            }
        finally:
            if is_cuda_alike():
                try:
                    torch.cuda.synchronize()
                except Exception as e:
                    logger.warning(
                        "DFLASH true partial proof-request synchronize before restore failed: %s",
                        e,
                    )
            for mapping, proof_req_pool_index, saved in reversed(saved_mamba_rows):
                mapping[proof_req_pool_index] = saved
            for proof_req_pool_index, end, saved in saved_req_rows:
                req_to_token[proof_req_pool_index, :end] = saved
            req_to_token_pool.free_slots = saved_free_slots
            allocator.restore_state(allocator_state)
            if mamba_pool is not None and saved_mamba_free_slots is not None:
                mamba_pool.free_slots = saved_mamba_free_slots
            restore_report["intermediate_cache_restore"] = (
                self._restore_true_partial_intermediate_cache(
                    batch=batch,
                    backup=intermediate_cache_backup,
                )
            )
            try:
                setattr(metadata_holder, "forward_metadata", saved_forward_metadata)
                restore_report["forward_metadata_restore"] = {
                    "present": saved_forward_metadata is not None,
                    "restored": True,
                }
            except Exception as e:
                logger.warning(
                    "DFLASH true partial proof-request forward-metadata restore failed: %s",
                    e,
                )
                restore_report["forward_metadata_restore"] = {
                    "present": saved_forward_metadata is not None,
                    "restored": False,
                    "error": str(e),
                }
            _append_recurrent_snapshot("after_restore")

    def _run_true_partial_verify_shadow_forward(
        self,
        *,
        batch: ScheduleBatch,
        model_worker_batch: ModelWorkerBatch,
        partial_spec: DFlashVerifyInput,
        partial_width: int,
        forward_kwargs: dict,
    ):
        """Run the proof forward with independent KV slots, then restore state."""

        from unifyinfer.traces.dflash_true_partial_shadow import (
            build_dflash_true_partial_shadow_forward_plan,
        )

        bs = batch.batch_size()
        prefix_lens_cpu = [int(item) for item in batch.seq_lens_cpu.tolist()]
        shadow_plan = build_dflash_true_partial_shadow_forward_plan(
            prefix_lens=prefix_lens_cpu,
            full_width=int(getattr(model_worker_batch.spec_info, "draft_token_num", 0)),
            partial_width=int(partial_width),
            page_size=int(self.page_size),
            allocator_class=batch.token_to_kv_pool_allocator.__class__.__module__
            + "."
            + batch.token_to_kv_pool_allocator.__class__.__name__,
        )
        if not shadow_plan["eligible"]:
            logger.warning(
                "DFLASH true partial shadow forward rejected: %s",
                shadow_plan.get("rejection_reason"),
            )
            return None, shadow_plan

        allocator = batch.token_to_kv_pool_allocator
        req_to_token = batch.req_to_token_pool.req_to_token
        allocator_state = allocator.backup_state()
        saved_req_to_token: list[tuple[int, int, torch.Tensor]] = []
        try:
            if self.page_size == 1:
                shadow_cache_loc = allocator.alloc(bs * int(partial_width))
            else:
                end_offset = batch.seq_lens + int(partial_width)
                end_offset_cpu = batch.seq_lens_cpu + int(partial_width)
                last_loc = get_last_loc(
                    req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                )
                shadow_cache_loc = allocator.alloc_extend(
                    batch.seq_lens,
                    batch.seq_lens_cpu,
                    end_offset,
                    end_offset_cpu,
                    last_loc,
                    bs * int(partial_width),
                )
            if shadow_cache_loc is None:
                shadow_plan = {
                    **shadow_plan,
                    "eligible": False,
                    "rejection_reason": "shadow_kv_slot_allocation_failed",
                }
                logger.warning("DFLASH true partial shadow forward rejected: OOM")
                return None, shadow_plan

            for req_pool_index, prefix_len in zip(
                batch.req_pool_indices.detach().cpu().tolist(),
                prefix_lens_cpu,
                strict=True,
            ):
                end = int(prefix_len) + int(partial_width)
                req_index = int(req_pool_index)
                saved_req_to_token.append(
                    (req_index, end, req_to_token[req_index, :end].detach().clone())
                )

            end_offset = batch.seq_lens + int(partial_width)
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                req_to_token,
                batch.seq_lens,
                end_offset,
                shadow_cache_loc,
                bs,
            )
            partial_worker_batch = replace(
                model_worker_batch,
                input_ids=partial_spec.draft_token,
                out_cache_loc=shadow_cache_loc,
                spec_info=partial_spec,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            partial_result = self.target_worker.forward_batch_generation(
                partial_worker_batch,
                is_verify=True,
                **forward_kwargs,
            )
            if is_cuda_alike():
                torch.cuda.synchronize()
            return partial_result, {
                **shadow_plan,
                "allocated_shadow_slots": int(shadow_cache_loc.numel()),
            }
        finally:
            if is_cuda_alike():
                try:
                    torch.cuda.synchronize()
                except Exception as e:
                    logger.warning(
                        "DFLASH true partial shadow synchronize before restore failed: %s",
                        e,
                    )
            for req_index, end, saved in saved_req_to_token:
                req_to_token[req_index, :end] = saved
            allocator.restore_state(allocator_state)

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise RuntimeError(
                "Invariant broken: DFLASH batch requested return_logprob, but scheduler should have rejected this request."
            )

        if isinstance(batch, ModelWorkerBatch):
            # Should not happen for spec-v1 (non-overlap) scheduling, but keep a sane fallback.
            return self.target_worker.forward_batch_generation(batch, **kwargs)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, **kwargs
            )
            logits_output, next_token_ids = (
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DFLASH requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            if (
                model_worker_batch.extend_seq_lens is None
                or model_worker_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "DFLASH expected extend_seq_lens / extend_prefix_lens to be populated in extend mode, but got None."
                )

            # Materialize the prompt tokens into the draft KV cache immediately. This is required
            # for radix cache support, since the scheduler may update radix after prefill returns.
            device = next_token_ids.device

            def _to_int32_device_tensor(x, *, device=device):
                if isinstance(x, torch.Tensor):
                    if x.device != device:
                        x = x.to(device, non_blocking=True)
                    return x if x.dtype == torch.int32 else x.to(torch.int32)
                return torch.tensor(x, dtype=torch.int32, device=device)

            extend_seq_lens = _to_int32_device_tensor(
                model_worker_batch.extend_seq_lens
            )
            draft_input = DFlashDraftInput(
                verified_id=next_token_ids.to(torch.int64),
                target_hidden=logits_output.hidden_states,
                ctx_lens=extend_seq_lens,
                draft_seq_lens=(
                    torch.zeros_like(extend_seq_lens)
                    if self.use_compact_draft_cache
                    else _to_int32_device_tensor(model_worker_batch.extend_prefix_lens)
                ),
            )
            self._append_target_hidden_to_draft_kv(batch, draft_input)
            batch.spec_info = draft_input

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        # Decode / target-verify stage.
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInput):
            raise RuntimeError(
                "DFLASH decode requires DFlashDraftInput state on the running batch. "
                "This usually means the request did not complete the prefill stage."
            )

        _profile_total = self._dflash_profile_start()
        self._prepare_for_speculative_decoding(batch, draft_input)
        _profile_prepare_ms = self._dflash_profile_elapsed_ms(_profile_total)

        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info
        assert isinstance(verify_input, DFlashVerifyInput)
        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )
        pre_verify_recurrent_seed = (
            self._backup_true_partial_recurrent_seed(
                batch=batch,
                bs=batch.batch_size(),
            )
            if (
                self._true_partial_proof_request_forward_enabled()
                and self._supports_true_partial_verify_capture(batch)
                and batch.batch_size() == 1
            )
            else None
        )

        _profile_phase = self._dflash_profile_start()
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True, **kwargs
        )
        _profile_target_verify_ms = self._dflash_profile_elapsed_ms(_profile_phase)
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )
        self._maybe_capture_true_partial_verify_forward(
            batch=batch,
            model_worker_batch=model_worker_batch,
            verify_input=verify_input,
            logits_output=logits_output,
            forward_kwargs=kwargs,
            pre_verify_recurrent_seed=pre_verify_recurrent_seed,
        )

        _profile_phase = self._dflash_profile_start()
        _verify_profile = {} if self._dflash_profile_enabled else None
        (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
            profile=_verify_profile,
        )
        _profile_verify_update_ms = self._dflash_profile_elapsed_ms(_profile_phase)
        if _verify_profile is not None:
            _verify_profile_total_ms = sum(float(v) for v in _verify_profile.values())
            self._dflash_profile_log(
                "DFLASH profile verify_detail: step=%d bs=%d block_size=%d "
                "logit_adjust_ms=%.3f accept_compute_ms=%.3f d2h_pack_ms=%.3f "
                "request_update_ms=%.3f tensor_build_ms=%.3f "
                "kv_free_compact_ms=%.3f req_kv_accounting_ms=%.3f "
                "req_to_token_update_ms=%.3f seq_lens_update_ms=%.3f "
                "hidden_slice_ms=%.3f hidden_clear_ms=%.3f "
                "accounted_ms=%.3f outer_verify_ms=%.3f commit_lens=%s",
                self._dflash_profile_step,
                batch.batch_size(),
                self.block_size,
                _verify_profile.get("logit_adjust_ms", 0.0),
                _verify_profile.get("accept_compute_ms", 0.0),
                _verify_profile.get("d2h_pack_ms", 0.0),
                _verify_profile.get("request_update_ms", 0.0),
                _verify_profile.get("tensor_build_ms", 0.0),
                _verify_profile.get("kv_free_compact_ms", 0.0),
                _verify_profile.get("req_kv_accounting_ms", 0.0),
                _verify_profile.get("req_to_token_update_ms", 0.0),
                _verify_profile.get("seq_lens_update_ms", 0.0),
                _verify_profile.get("hidden_slice_ms", 0.0),
                _verify_profile.get("hidden_clear_ms", 0.0),
                _verify_profile_total_ms,
                _profile_verify_update_ms,
                commit_lens.detach().cpu().tolist(),
            )
        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            _profile_phase = self._dflash_profile_start()
            self._update_target_mamba_state_after_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )
            _profile_mamba_ms = self._dflash_profile_elapsed_ms(_profile_phase)
        else:
            _profile_mamba_ms = 0.0

        # Update draft state for the next iteration. Also materialize the committed verify tokens
        # into the draft KV cache immediately so radix cache entries are safe to reuse.
        draft_input.verified_id = new_verified_id
        draft_input.target_hidden = next_target_hidden
        draft_input.ctx_lens = commit_lens
        _profile_phase = self._dflash_profile_start()
        self._append_target_hidden_to_draft_kv(batch, draft_input)
        _profile_kv_after_ms = self._dflash_profile_elapsed_ms(_profile_phase)
        batch.spec_info = draft_input
        batch.forward_mode = ForwardMode.DECODE

        num_accepted_tokens = sum(accept_length_per_req_cpu)
        _profile_total_ms = self._dflash_profile_elapsed_ms(_profile_total)
        self._dflash_profile_log(
            "DFLASH profile decode: step=%d bs=%d block_size=%d "
            "prepare_ms=%.3f target_verify_forward_ms=%.3f "
            "verify_accept_update_ms=%.3f mamba_commit_ms=%.3f "
            "kv_after_verify_ms=%.3f total_ms=%.3f accepted_tokens=%d commit_lens=%s",
            self._dflash_profile_step,
            batch.batch_size(),
            self.block_size,
            _profile_prepare_ms,
            _profile_target_verify_ms,
            _profile_verify_update_ms,
            _profile_mamba_ms,
            _profile_kv_after_ms,
            _profile_total_ms,
            num_accepted_tokens,
            commit_lens.detach().cpu().tolist(),
        )
        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info(
                "DFLASH verify completed. accept_length_per_req=%s",
                accept_length_per_req_cpu,
            )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=new_verified_id,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )
