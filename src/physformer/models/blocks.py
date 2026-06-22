from __future__ import annotations

import atexit
from collections import Counter
import math
import os
from typing import Optional

import torch

import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func as _flash_attn_func

    _FLASH_ATTN_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover - depends on optional CUDA extension install.
    _flash_attn_func = None
    _FLASH_ATTN_IMPORT_ERROR = exc


_ATTENTION_LOGGED: set[str] = set()
_ATTENTION_BACKEND_COUNTS: Counter[str] = Counter()
_ATTENTION_BACKEND_EXAMPLES: dict[str, str] = {}
_ATTENTION_SUMMARY_REGISTERED = False


def _attention_debug_enabled() -> bool:
    value = os.environ.get("PHYSFORMER_SDPA_DEBUG", os.environ.get("JMT4D_SDPA_DEBUG", "0")).strip().lower()
    return value not in ("", "0", "false", "no", "off")


def _attention_log_once(key: str, msg: str, *, debug_only: bool = False) -> None:
    if debug_only and not _attention_debug_enabled():
        return
    if key in _ATTENTION_LOGGED:
        return
    _ATTENTION_LOGGED.add(key)
    rank = os.environ.get("RANK", "0")
    print(f"[attention][rank={rank}] {msg}", flush=True)


def _attention_chunk_size() -> int:
    raw = os.environ.get("PHYSFORMER_ATTENTION_CHUNK_SIZE", "512").strip()
    try:
        value = int(raw)
    except ValueError:
        return 512
    return max(1, value)


def _record_attention_backend(name: str, q: torch.Tensor, *, attn_mask: Optional[torch.Tensor]) -> None:
    _ATTENTION_BACKEND_COUNTS[str(name)] += 1
    if str(name) not in _ATTENTION_BACKEND_EXAMPLES:
        mask_kind = "masked" if attn_mask is not None else "unmasked"
        _ATTENTION_BACKEND_EXAMPLES[str(name)] = f"{mask_kind} dtype={q.dtype} q={tuple(q.shape)}"


def _print_attention_backend_summary() -> None:
    if not _attention_debug_enabled() or not _ATTENTION_BACKEND_COUNTS:
        return
    rank = os.environ.get("RANK", "0")
    total = sum(_ATTENTION_BACKEND_COUNTS.values())
    print(f"[attention-summary][rank={rank}] total_calls={total}", flush=True)
    for name in sorted(_ATTENTION_BACKEND_COUNTS):
        example = _ATTENTION_BACKEND_EXAMPLES.get(name, "")
        print(
            f"[attention-summary][rank={rank}] {name}={_ATTENTION_BACKEND_COUNTS[name]} example=({example})",
            flush=True,
        )


def _ensure_attention_summary_registered() -> None:
    global _ATTENTION_SUMMARY_REGISTERED
    if _ATTENTION_SUMMARY_REGISTERED:
        return
    atexit.register(_print_attention_backend_summary)
    _ATTENTION_SUMMARY_REGISTERED = True


def _configure_torch_sdpa() -> None:
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None:
        return
    settings = {
        "enable_flash_sdp": True,
        "enable_mem_efficient_sdp": True,
        "enable_math_sdp": False,
    }
    for name, value in settings.items():
        fn = getattr(cuda_backend, name, None)
        if callable(fn):
            fn(value)


_configure_torch_sdpa()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Keep output dtype stable under AMP: bf16/fp16 * fp32 promotes to fp32,
        # which can make PyTorch fast CUDA SDPA unavailable.
        return x.to(dtype) * self.weight.to(dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if shift.dtype != x.dtype:
        shift = shift.to(dtype=x.dtype)
    if scale.dtype != x.dtype:
        scale = scale.to(dtype=x.dtype)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.drop(hidden))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = float(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        dropout_p: float,
        is_causal: bool,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if _attention_debug_enabled():
            _ensure_attention_summary_registered()

        def _has_sdp_kernel() -> bool:
            cuda_backend = getattr(torch.backends, "cuda", None)
            return cuda_backend is not None and hasattr(cuda_backend, "sdp_kernel")

        def _fast_attention_unavailable_detail(cause: Optional[BaseException] = None) -> str:
            detail = (
                f"attention_device={q.device} attention_dtype={q.dtype} attention_shape={tuple(q.shape)}. "
            )
            if _flash_attn_func is None:
                detail += (
                    " External flash-attn is not importable"
                    f" ({type(_FLASH_ATTN_IMPORT_ERROR).__name__}: {_FLASH_ATTN_IMPORT_ERROR})."
                )
            if not q.is_cuda:
                detail += f" The attention tensor is on {q.device}, not CUDA."
            elif q.dtype not in (torch.float16, torch.bfloat16):
                detail += " Fast CUDA SDPA requires float16 or bfloat16 attention tensors."
            elif not _has_sdp_kernel():
                detail += " This PyTorch build does not expose torch.backends.cuda.sdp_kernel."
            if cause is None:
                return detail
            return f"{detail} Last fast-attention error: {type(cause).__name__}: {cause}"

        def _try_sdpa_with_sdp_kernel(*, flash: bool, mem_efficient: bool, math_backend: bool) -> torch.Tensor:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=bool(flash),
                enable_mem_efficient=bool(mem_efficient),
                enable_math=bool(math_backend),
            ):
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
                )

        def _try_external_flash_attn() -> Optional[torch.Tensor]:
            if _flash_attn_func is None:
                detail = f"{type(_FLASH_ATTN_IMPORT_ERROR).__name__}: {_FLASH_ATTN_IMPORT_ERROR}"
                _attention_log_once(
                    "external-flash-import-missing",
                    "External flash-attn is not importable; falling back to PyTorch SDPA or math/chunked attention. "
                    f"Install the README prebuilt flash-attn wheel if you want the preferred path. ({detail})",
                )
                return None
            if attn_mask is not None:
                _attention_log_once(
                    "external-flash-mask-fallback",
                    "External flash-attn is installed, but this attention call has a padding mask; "
                    "using PyTorch SDPA or math/chunked fallback for masked attention.",
                )
                return None

            try:
                # flash_attn_func expects (B, S, H, Hd); the model stores (B, H, S, Hd).
                q_fa = q.transpose(1, 2).contiguous()
                k_fa = k.transpose(1, 2).contiguous()
                v_fa = v.transpose(1, 2).contiguous()
                out = _flash_attn_func(q_fa, k_fa, v_fa, dropout_p=dropout_p, causal=is_causal)
                _attention_log_once(
                    "external-flash-active",
                    "Using external flash-attn as the preferred attention backend "
                    f"(dtype={q.dtype}, q={tuple(q.shape)}).",
                )
                _record_attention_backend("external_flash_attn", q, attn_mask=attn_mask)
                return out.transpose(1, 2).contiguous()
            except Exception as exc:
                _attention_log_once(
                    "external-flash-runtime-fallback",
                    "External flash-attn was importable but failed for this attention shape; "
                    "falling back to PyTorch SDPA or math/chunked attention. "
                    f"reason={type(exc).__name__}: {exc}",
                )
                return None

        def _try_math_sdpa() -> torch.Tensor:
            if _has_sdp_kernel():
                return _try_sdpa_with_sdp_kernel(flash=False, mem_efficient=False, math_backend=True)
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
            )

        def _chunked_math_attention() -> torch.Tensor:
            seq_len = int(q.shape[-2])
            chunk_size = _attention_chunk_size()
            scale = 1.0 / math.sqrt(float(q.shape[-1]))
            k_t = k.transpose(-2, -1)
            out_chunks: list[torch.Tensor] = []
            key_positions = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len) if is_causal else None

            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                q_chunk = q[..., start:end, :]
                scores = torch.matmul(q_chunk, k_t) * scale
                bool_mask: Optional[torch.Tensor] = None

                if attn_mask is not None:
                    mask = attn_mask
                    if mask.shape[-2] == seq_len:
                        mask = mask[..., start:end, :]
                    if mask.dtype == torch.bool:
                        bool_mask = mask
                        all_masked = ~mask.any(dim=-1, keepdim=True)
                        scores = scores.masked_fill(~mask, float("-inf"))
                        scores = scores.masked_fill(all_masked, 0.0)
                    else:
                        scores = scores + mask

                causal_mask: Optional[torch.Tensor] = None
                if key_positions is not None:
                    query_positions = torch.arange(start, end, device=q.device).view(1, 1, end - start, 1)
                    causal_mask = key_positions <= query_positions
                    scores = scores.masked_fill(~causal_mask, float("-inf"))

                weights = torch.softmax(scores.float(), dim=-1).to(dtype=v.dtype)
                if bool_mask is not None:
                    weights = weights.masked_fill(~bool_mask, 0.0)
                if causal_mask is not None:
                    weights = weights.masked_fill(~causal_mask, 0.0)
                if dropout_p > 0.0:
                    weights = F.dropout(weights, p=float(dropout_p), training=True)
                out_chunks.append(torch.matmul(weights, v))

            return torch.cat(out_chunks, dim=-2).to(dtype=q.dtype)

        def _math_or_chunked_fallback(cause: Optional[BaseException] = None) -> torch.Tensor:
            detail = _fast_attention_unavailable_detail(cause)
            _attention_log_once(
                "math-chunked-fallback-warning",
                "CUDA fast attention unavailable for this call; using math/chunked attention fallback. "
                "This is expected to be much slower and may use more memory than external flash-attn, "
                "PyTorch flash SDPA, or PyTorch memory-efficient CUDA SDPA. "
                f"{detail}",
            )
            try:
                out = _try_math_sdpa()
                _attention_log_once(
                    "pytorch-math-sdpa-active",
                    "Using PyTorch math SDPA fallback "
                    f"(dtype={q.dtype}, q={tuple(q.shape)}).",
                )
                _record_attention_backend("pytorch_math_sdpa", q, attn_mask=attn_mask)
                return out
            except Exception as math_exc:
                if q.is_cuda:
                    torch.cuda.empty_cache()
                _attention_log_once(
                    "chunked-math-attention-active",
                    "PyTorch math SDPA also failed; using chunked manual attention fallback "
                    f"(chunk_size={_attention_chunk_size()}, dtype={q.dtype}, q={tuple(q.shape)}). "
                    f"reason={type(math_exc).__name__}: {math_exc}",
                )
                out = _chunked_math_attention()
                _record_attention_backend("chunked_math_attention", q, attn_mask=attn_mask)
                return out

        external_flash_eligible = q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
        if external_flash_eligible:
            out = _try_external_flash_attn()
            if out is not None:
                return out

        torch_fast_eligible = (
            q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and _has_sdp_kernel()
        )
        if not torch_fast_eligible:
            return _math_or_chunked_fallback()

        try:
            if _attention_debug_enabled():
                try:
                    out = _try_sdpa_with_sdp_kernel(flash=True, mem_efficient=False, math_backend=False)
                    _attention_log_once(
                        "pytorch-flash-sdpa-active",
                        "Using PyTorch flash SDPA fallback "
                        f"(dtype={q.dtype}, q={tuple(q.shape)}).",
                    )
                    _record_attention_backend("pytorch_flash_sdpa", q, attn_mask=attn_mask)
                    return out
                except Exception:
                    out = _try_sdpa_with_sdp_kernel(flash=False, mem_efficient=True, math_backend=False)
                    _attention_log_once(
                        "pytorch-mem-efficient-sdpa-active",
                        "Using PyTorch memory-efficient SDPA fallback "
                        f"(PyTorch flash SDPA was unavailable; dtype={q.dtype}, q={tuple(q.shape)}).",
                    )
                    _record_attention_backend("pytorch_mem_efficient_sdpa", q, attn_mask=attn_mask)
                    return out

            _attention_log_once(
                "pytorch-sdpa-fallback",
                "Using PyTorch SDPA fallback with flash and memory-efficient CUDA kernels enabled "
                "(math fallback disabled unless fast SDPA fails). Pass --attention-debug to print the exact PyTorch SDPA backend.",
            )
            out = _try_sdpa_with_sdp_kernel(flash=True, mem_efficient=True, math_backend=False)
            _record_attention_backend("pytorch_fast_sdpa", q, attn_mask=attn_mask)
            return out
        except Exception as exc:
            return _math_or_chunked_fallback(exc)

    def forward(self, x: torch.Tensor, rope, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, S, Hd)

        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None:
            q = rope(q)
            k = rope(k)

        attn_mask = None
        if src_key_padding_mask is not None:
            if src_key_padding_mask.ndim != 2 or src_key_padding_mask.shape != (bsz, seq_len):
                raise ValueError(f"src_key_padding_mask must be (B,S)={bsz, seq_len}, got {tuple(src_key_padding_mask.shape)}")
            # RenderFormer-style semantics: True means "valid / keep".
            keep = src_key_padding_mask.to(device=q.device, dtype=torch.bool).view(bsz, 1, 1, seq_len)
            attn_mask = keep.expand(bsz, self.num_heads, 1, seq_len)  # (B,H,1,S)

        # scaled_dot_product_attention expects (B, H, S, Hd)
        dropout_p = self.attn_drop if self.training else 0.0
        x = self._sdpa(q, k, v, dropout_p=float(dropout_p), is_causal=False, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(bsz, seq_len, dim)
        x = self.proj(x)
        return self.proj_drop(x)


class DiTBlock(nn.Module):
    """
    DiT-style transformer block with AdaLN (shift/scale/gates).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop, qkv_bias=True, qk_norm=True
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        rope=None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope=rope,
            src_key_padding_mask=src_key_padding_mask,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)
