import torch
import triton
import triton.language as tl


@triton.jit
def _ggml_gdn_row_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    state_ptr,
    state_indices_ptr,
    out_ptr,
    scale: tl.constexpr,
    T: tl.constexpr,
    H_K: tl.constexpr,
    H_V: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    h = tl.program_id(0)
    j = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)

    kv_repeat = H_V // H_K
    kv_h = h // kv_repeat
    state_index = tl.load(state_indices_ptr).to(tl.int64)
    state_base = state_ptr + (
        state_index * H_V * D * D + h * D * D + j * D + offs
    ).to(tl.int64)

    s = tl.load(state_base, mask=offs < D, other=0.0).to(tl.float32)

    for t in tl.range(0, T):
        k_vec = tl.load(k_ptr + t * H_K * D + kv_h * D + offs, mask=offs < D).to(
            tl.float32
        )
        q_vec = tl.load(q_ptr + t * H_K * D + kv_h * D + offs, mask=offs < D).to(
            tl.float32
        )
        g_val = tl.load(g_ptr + t * H_V + h).to(tl.float32)
        beta_val = tl.load(beta_ptr + t * H_V + h).to(tl.float32)
        v_val = tl.load(v_ptr + t * H_V * D + h * D + j).to(tl.float32)

        s = tl.exp(g_val) * s
        kv_col = tl.sum(s * k_vec, axis=0)
        delta_col = (v_val - kv_col) * beta_val
        s = s + k_vec * delta_col
        attn_col = tl.sum(s * q_vec, axis=0) * scale
        tl.store(out_ptr + t * H_V * D + h * D + j, attn_col)

    tl.store(state_base, s, mask=offs < D)


def ggml_gdn_row_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run the ggml-style per-value-row GDN recurrence for one prefill sequence.

    The kernel updates ``initial_state[initial_state_indices[0]]`` in place, matching
    the existing FLA chunk path's GPU-side state-pool update contract.  It returns
    only the output tensor because this guarded path is used only when prefix-cache
    tracking does not need FLA's intermediate ``h`` chunk-boundary states.
    """
    _, seq_len, num_k_heads, head_dim = q.shape
    num_v_heads = v.shape[2]
    out = torch.empty_like(v)
    _ggml_gdn_row_kernel[(num_v_heads, head_dim)](
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        initial_state_indices,
        out,
        scale,
        seq_len,
        num_k_heads,
        num_v_heads,
        head_dim,
        triton.next_power_of_2(head_dim),
        num_warps=1,
    )
    return out


def can_use_ggml_gdn_row_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    min_tokens: int,
    has_mamba_track_mask: bool,
    g_is_chunk_cumsum: bool,
) -> bool:
    if has_mamba_track_mask or g_is_chunk_cumsum:
        return False
    if initial_state is None or initial_state_indices is None:
        return False
    if not (
        q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and g.is_cuda
        and beta.is_cuda
        and initial_state.is_cuda
        and initial_state_indices.is_cuda
    ):
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if g.ndim != 3 or beta.ndim != 3:
        return False
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        return False
    if q.shape[1] < min_tokens:
        return False
    if q.shape != k.shape:
        return False
    if q.shape[1] != v.shape[1] or q.shape[-1] != v.shape[-1]:
        return False
    if q.shape[-1] > 256:
        return False
    if v.shape[2] % q.shape[2] != 0:
        return False
    if g.shape != beta.shape or g.shape != (1, q.shape[1], v.shape[2]):
        return False
    if initial_state.ndim != 4:
        return False
    if initial_state.shape[1:] != (v.shape[2], v.shape[3], k.shape[3]):
        return False
    if initial_state_indices.numel() != 1:
        return False
    return True
