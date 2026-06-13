from __future__ import annotations

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


def _attention_debug_enabled() -> bool:
    value = os.environ.get("JMT4D_SDPA_DEBUG", "0").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def _attention_log_once(key: str, msg: str, *, debug_only: bool = False) -> None:
    if debug_only and not _attention_debug_enabled():
        return
    if key in _ATTENTION_LOGGED:
        return
    _ATTENTION_LOGGED.add(key)
    rank = os.environ.get("RANK", "0")
    print(f"[attention][rank={rank}] {msg}", flush=True)


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

        def _raise_fast_sdpa_unavailable(cause: Optional[BaseException] = None) -> None:
            detail = (
                "CUDA fast attention unavailable: this demo requires external flash-attn, "
                "PyTorch flash SDPA, or PyTorch memory-efficient CUDA SDPA, and exits instead "
                "of using math/chunked fallback. "
                f"attention_device={q.device} attention_dtype={q.dtype} attention_shape={tuple(q.shape)}. "
                "Check that CUDA is available, run with --device cuda --amp bf16 or --amp fp16, "
                "and use a CUDA PyTorch wheel on a supported NVIDIA GPU."
            )
            if not q.is_cuda:
                detail += f" The attention tensor is on {q.device}, not CUDA."
            elif q.dtype not in (torch.float16, torch.bfloat16):
                detail += " Fast CUDA SDPA requires float16 or bfloat16 attention tensors."
            elif not hasattr(torch.backends.cuda, "sdp_kernel"):
                detail += " This PyTorch build does not expose torch.backends.cuda.sdp_kernel."
            if cause is None:
                raise RuntimeError(detail)
            raise RuntimeError(detail) from cause

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
                    "External flash-attn is not importable; falling back to PyTorch SDPA. "
                    f"Install the README prebuilt flash-attn wheel if you want the preferred path. ({detail})",
                )
                return None
            if attn_mask is not None:
                _attention_log_once(
                    "external-flash-mask-fallback",
                    "External flash-attn is installed, but this attention call has a padding mask; "
                    "using PyTorch SDPA for masked attention.",
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
                return out.transpose(1, 2).contiguous()
            except Exception as exc:
                _attention_log_once(
                    "external-flash-runtime-fallback",
                    "External flash-attn was importable but failed for this attention shape; "
                    "falling back to PyTorch SDPA. "
                    f"reason={type(exc).__name__}: {exc}",
                )
                return None

        if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16) or not hasattr(torch.backends.cuda, "sdp_kernel"):
            _raise_fast_sdpa_unavailable()

        out = _try_external_flash_attn()
        if out is not None:
            return out

        try:
            if _attention_debug_enabled():
                try:
                    out = _try_sdpa_with_sdp_kernel(flash=True, mem_efficient=False, math_backend=False)
                    _attention_log_once(
                        "pytorch-flash-sdpa-active",
                        "Using PyTorch flash SDPA fallback "
                        f"(dtype={q.dtype}, q={tuple(q.shape)}).",
                    )
                    return out
                except Exception:
                    out = _try_sdpa_with_sdp_kernel(flash=False, mem_efficient=True, math_backend=False)
                    _attention_log_once(
                        "pytorch-mem-efficient-sdpa-active",
                        "Using PyTorch memory-efficient SDPA fallback "
                        f"(PyTorch flash SDPA was unavailable; dtype={q.dtype}, q={tuple(q.shape)}).",
                    )
                    return out

            _attention_log_once(
                "pytorch-sdpa-fallback",
                "Using PyTorch SDPA fallback with flash and memory-efficient CUDA kernels enabled "
                "(math fallback disabled). Pass --attention-debug to print the exact PyTorch SDPA backend.",
            )
            return _try_sdpa_with_sdp_kernel(flash=True, mem_efficient=True, math_backend=False)
        except Exception as exc:
            _raise_fast_sdpa_unavailable(exc)

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
