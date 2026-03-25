from __future__ import annotations

import gzip
import os
import time
from contextlib import contextmanager, nullcontext


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _profile_dir() -> str | None:
    d = os.environ.get("VLLM_OMNI_TORCH_PROFILE_DIR")
    if not d:
        return None
    return os.path.abspath(d)


class _WorkerTorchProfiler:
    def __init__(self) -> None:
        self.enabled = _profile_dir() is not None
        self._prof = None
        self._step = 0
        self._max_steps = _env_int("VLLM_OMNI_TORCH_PROFILE_MAX_STEPS", 50)
        self._with_stack = _env_flag("VLLM_OMNI_TORCH_PROFILE_WITH_STACK", False)
        self._record_shapes = _env_flag("VLLM_OMNI_TORCH_PROFILE_RECORD_SHAPES", False)
        self._profile_memory = _env_flag("VLLM_OMNI_TORCH_PROFILE_MEMORY", False)
        self._with_flops = _env_flag("VLLM_OMNI_TORCH_PROFILE_FLOPS", False)
        self._use_gzip = _env_flag("VLLM_OMNI_TORCH_PROFILE_GZIP", True)

    def _make_trace_path(self) -> str:
        d = _profile_dir()
        assert d is not None
        os.makedirs(d, exist_ok=True)
        pid = os.getpid()
        ts = int(time.time())
        rank = os.environ.get("RANK", "0")
        return os.path.join(d, f"torchtrace_rank{rank}_pid{pid}_{ts}.json")

    def _start_if_needed(self):
        if not self.enabled or self._prof is not None:
            return
        try:
            import torch
            from torch.profiler import ProfilerActivity

            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)

            trace_path = self._make_trace_path()

            def _on_trace_ready(p):
                try:
                    p.export_chrome_trace(trace_path)
                    if self._use_gzip:
                        with open(trace_path, "rb") as f_in, gzip.open(trace_path + ".gz", "wb") as f_out:
                            f_out.writelines(f_in)
                        try:
                            os.remove(trace_path)
                        except OSError:
                            pass
                except Exception:
                    # Best-effort: profiling must never break inference.
                    pass

            self._prof = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=max(1, self._max_steps)),
                on_trace_ready=_on_trace_ready,
                with_stack=self._with_stack,
                record_shapes=self._record_shapes,
                profile_memory=self._profile_memory,
                with_flops=self._with_flops,
            )
            self._prof.start()
        except Exception:
            self._prof = None
            self.enabled = False

    def _stop_if_done(self):
        if self._prof is None:
            return
        if self._max_steps > 0 and self._step >= self._max_steps:
            try:
                self._prof.stop()
            except Exception:
                pass
            self._prof = None

    @contextmanager
    def step_ctx(self, step_name: str | None = None):
        if not self.enabled:
            yield
            return
        self._start_if_needed()
        if self._prof is None:
            yield
            return
        try:
            if step_name:
                try:
                    import torch

                    rf = getattr(torch.profiler, "record_function", None) or getattr(
                        torch.autograd.profiler, "record_function", None
                    )
                    if rf is None:
                        yield
                    else:
                        with rf(step_name):
                            yield
                except Exception:
                    yield
            else:
                yield
        finally:
            try:
                self._prof.step()
            except Exception:
                pass
            self._step += 1
            self._stop_if_done()


_SINGLETON: _WorkerTorchProfiler | None = None


def _get() -> _WorkerTorchProfiler:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = _WorkerTorchProfiler()
    return _SINGLETON


def worker_profiler_step(step_name: str | None = None):
    """Return a context manager that steps a worker-local torch.profiler.

    Enable by setting `VLLM_OMNI_TORCH_PROFILE_DIR=/path`.
    """
    prof = _get()
    if not prof.enabled:
        return nullcontext()
    return prof.step_ctx(step_name=step_name)

