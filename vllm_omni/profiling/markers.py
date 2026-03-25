from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _markers_enabled() -> bool:
    # Keep default off to avoid any overhead on production runs.
    return _env_flag("VLLM_OMNI_PROFILE_MARKERS", default=False)


@contextmanager
def prof_range(name: str):
    """A low-overhead range marker for torch.profiler / chrome traces.

    Enabled only when `VLLM_OMNI_PROFILE_MARKERS=1`.
    """
    if not _markers_enabled():
        yield
        return
    try:
        import torch

        record_fn = getattr(torch.profiler, "record_function", None)
        if record_fn is None:
            record_fn = getattr(torch.autograd.profiler, "record_function", None)
        if record_fn is None:
            yield
            return
        with record_fn(name):
            yield
    except Exception:
        # Never fail inference due to optional profiling.
        yield


@contextmanager
def nvtx_range(name: str):
    """An NVTX range for Nsight Systems.

    Enabled only when `VLLM_OMNI_PROFILE_MARKERS=1`.
    """
    if not _markers_enabled():
        yield
        return
    try:
        import torch

        if not torch.cuda.is_available():
            yield
            return
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    except Exception:
        yield


@contextmanager
def combined_range(name: str):
    """Emit both torch.profiler and NVTX ranges."""
    if not _markers_enabled():
        yield
        return
    with prof_range(name), nvtx_range(name):
        yield


def cuda_profiler_api_enabled() -> bool:
    """Enable CUDA profiler start/stop for `nsys --capture-range=cudaProfilerApi`."""
    return _env_flag("VLLM_OMNI_NSYS_CAPTURE", default=False)


@contextmanager
def cuda_profiler_api_range(name: str):
    """Start/stop CUDA profiler for `nsys --capture-range=cudaProfilerApi`.

    Note: CUDA profiler is process-global. Only use with 1 inflight request.
    """
    if not cuda_profiler_api_enabled():
        yield
        return
    try:
        import torch

        if not torch.cuda.is_available():
            yield
            return
        try:
            torch.cuda.nvtx.range_push(f"cudaProfilerApi:{name}")
        except Exception:
            pass
        torch.cuda.profiler.start()
        try:
            yield
        finally:
            try:
                torch.cuda.profiler.stop()
            finally:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass
    except Exception:
        yield


def null_or_range(name: str):
    """Convenience helper to keep call sites clean."""
    return combined_range(name) if _markers_enabled() else nullcontext()

