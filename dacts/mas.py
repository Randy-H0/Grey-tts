"""Monotonic alignment search (Glow-TTS).

CUDA tensors use a Triton kernel (one program per utterance, rows processed in order with a
block barrier); results match the numba CPU reference exactly. This avoids a GPU->CPU copy
and sync every training step.
"""
import numba
import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None


@numba.njit(nogil=True, cache=True)
def _path_single(path, value, t_y, t_x):
    neg = -1e9
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = neg if x == y else value[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else neg
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index -= 1


@numba.njit(nogil=True, parallel=True, cache=True)
def _path_batch(paths, values, t_ys, t_xs):
    for i in numba.prange(paths.shape[0]):
        _path_single(paths[i], values[i], t_ys[i], t_xs[i])


def _maximum_path_cpu(logp, mask):
    value = np.ascontiguousarray((logp.float() * mask).transpose(1, 2).cpu().numpy())
    path = np.zeros(value.shape, dtype=np.int32)
    t_x = mask[:, :, 0].sum(1).int().cpu().numpy()
    t_y = mask[:, 0, :].sum(1).int().cpu().numpy()
    _path_batch(path, value, t_y, t_x)
    return torch.from_numpy(path).to(logp.device).transpose(1, 2).float()


if triton is not None:

    @triton.jit
    def _mas_kernel(value_ptr, path_ptr, tx_ptr, ty_ptr, stride_b, stride_y, BLOCK_X: tl.constexpr):
        b = tl.program_id(0)
        t_x = tl.load(tx_ptr + b)
        t_y = tl.load(ty_ptr + b)
        val = value_ptr + b * stride_b
        xs = tl.arange(0, BLOCK_X)
        in_x = xs < t_x
        neg = -1e9

        # row 0: only x == 0 is reachable
        row0 = tl.load(val + xs, mask=in_x, other=neg)
        tl.store(val + xs, tl.where(xs == 0, row0, neg), mask=in_x)
        tl.debug_barrier()
        # forward pass: value[y, x] += max(value[y-1, x], value[y-1, x-1]); unreachable cells stay ~neg
        for y in range(1, t_y):
            prev = val + (y - 1) * stride_y
            stay = tl.load(prev + xs, mask=in_x, other=neg)
            adv = tl.load(prev + xs - 1, mask=in_x & (xs >= 1), other=neg)
            cur = tl.load(val + y * stride_y + xs, mask=in_x, other=0.0)
            tl.store(val + y * stride_y + xs, cur + tl.maximum(stay, adv), mask=in_x)
            tl.debug_barrier()

        # backtrack (scalar)
        pth = path_ptr + b * stride_b
        index = t_x - 1
        for j in range(0, t_y):
            y = t_y - 1 - j
            tl.store(pth + y * stride_y + index, 1)
            has_prev = y > 0
            v_stay = tl.load(val + (y - 1) * stride_y + index, mask=has_prev, other=0.0)
            v_adv = tl.load(val + (y - 1) * stride_y + index - 1, mask=has_prev & (index > 0), other=0.0)
            step = (index != 0) & ((index == y) | (v_stay < v_adv))
            index = tl.where(step, index - 1, index)


@torch.no_grad()
def maximum_path(logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """logp, mask: (B, Tx, Ty). Returns hard monotonic alignment (B, Tx, Ty) as float."""
    if triton is None or not logp.is_cuda:
        return _maximum_path_cpu(logp, mask)
    b, t_x, t_y = logp.shape
    value = logp.float().transpose(1, 2).contiguous()  # B, Ty, Tx
    path = torch.zeros(b, t_y, t_x, dtype=torch.int32, device=logp.device)
    tx = mask[:, :, 0].sum(1).int()
    ty = mask[:, 0, :].sum(1).int()
    block = triton.next_power_of_2(t_x)
    _mas_kernel[(b,)](value, path, tx, ty, value.stride(0), value.stride(1),
                      BLOCK_X=block, num_warps=max(1, min(16, block // 64)))
    return path.transpose(1, 2).float()
