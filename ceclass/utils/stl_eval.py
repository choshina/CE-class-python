"""
stlcgpp robustness at t=0: prefer full-batch vmap on GPU; optional multi-GPU
trace sharding; chunked fallback on CUDA OOM.
"""
from __future__ import annotations

import concurrent.futures
from typing import Callable, Optional, Sequence

import torch


def _is_cuda_oom(err: BaseException) -> bool:
    if hasattr(torch, "OutOfMemoryError") and isinstance(err, torch.OutOfMemoryError):
        return True
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom is not None and isinstance(err, oom):
        return True
    msg = str(err).lower()
    return "out of memory" in msg


def _resolve_eval_devices(
    traces: torch.Tensor,
    primary: torch.device,
    eval_devices: Optional[Sequence[torch.device]],
) -> tuple[torch.device, ...]:
    if eval_devices is not None:
        return tuple(eval_devices)
    if primary.type == "cuda" and torch.cuda.device_count() >= 2:
        return (torch.device("cuda:0"), torch.device("cuda:1"))
    return (primary,)


def _rob0_from_vmap(
    stl_formula: torch.nn.Module,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (rob0 tensor, full rob) for min/max over batch dim."""
    with torch.no_grad():
        rob = torch.vmap(stl_formula)(batch)
        rob0 = rob[:, 0] if rob.ndim > 1 else rob
    return rob0, rob


def _min_rob0_one_device_try_full_then_chunk(
    stl_formula: torch.nn.Module,
    traces_on_dev: torch.Tensor,
    chunk_size: Optional[int],
) -> float:
    """Min rob0 over traces; try full vmap first unless chunk_size is set."""
    n = traces_on_dev.shape[0]
    if n == 0:
        return 1e9

    if chunk_size is None:
        try:
            rob0, _ = _rob0_from_vmap(stl_formula, traces_on_dev)
            return rob0.min().item()
        except RuntimeError as e:
            if not _is_cuda_oom(e):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Chunked fallback (or forced chunk_size)
    cs = chunk_size if chunk_size is not None else max(1, n // 2)
    mins: list[float] = []
    while True:
        try:
            for start in range(0, n, cs):
                batch = traces_on_dev[start : start + cs]
                rob0, _ = _rob0_from_vmap(stl_formula, batch)
                mins.append(rob0.min().item())
            return min(mins)
        except RuntimeError as e:
            if not _is_cuda_oom(e):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if cs <= 1:
                raise
            cs = max(1, cs // 2)
            mins.clear()


def _max_rob0_one_device_try_full_then_chunk(
    stl_formula: torch.nn.Module,
    traces_on_dev: torch.Tensor,
    chunk_size: Optional[int],
) -> float:
    n = traces_on_dev.shape[0]
    if n == 0:
        return -1e9

    if chunk_size is None:
        try:
            rob0, _ = _rob0_from_vmap(stl_formula, traces_on_dev)
            return rob0.max().item()
        except RuntimeError as e:
            if not _is_cuda_oom(e):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    cs = chunk_size if chunk_size is not None else max(1, n // 2)
    maxes: list[float] = []
    while True:
        try:
            for start in range(0, n, cs):
                batch = traces_on_dev[start : start + cs]
                rob0, _ = _rob0_from_vmap(stl_formula, batch)
                maxes.append(rob0.max().item())
            return max(maxes)
        except RuntimeError as e:
            if not _is_cuda_oom(e):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if cs <= 1:
                raise
            cs = max(1, cs // 2)
            maxes.clear()


def min_rob0_vmap(
    make_stl: Callable[[torch.device], torch.nn.Module],
    traces: torch.Tensor,
    primary_device: torch.device,
    eval_devices: Optional[Sequence[torch.device]] = None,
    chunk_size: Optional[int] = None,
) -> float:
    """
    min_i rho(phi, trace_i) at t=0.

    - ``make_stl(dev)`` builds the stlcgpp module on ``dev`` (needed for multi-GPU).
    - Default ``eval_devices``: use ``cuda:0`` and ``cuda:1`` when two GPUs exist
      and ``primary_device`` is CUDA; otherwise only ``primary_device``.
    - Tries a full vmap on each shard first; on CUDA OOM, halves chunk size until it fits.
    """
    devices = _resolve_eval_devices(traces, primary_device, eval_devices)
    n = traces.shape[0]
    if n == 0:
        return 1e9

    if len(devices) == 1:
        dev = devices[0]
        stl = make_stl(dev)
        t = traces.to(dev)
        return _min_rob0_one_device_try_full_then_chunk(stl, t, chunk_size)

    # Multi-GPU: split batch along trace dimension
    n_dev = len(devices)
    sizes = [n // n_dev + (1 if i < n % n_dev else 0) for i in range(n_dev)]
    starts = [0]
    for s in sizes[:-1]:
        starts.append(starts[-1] + s)

    def _shard(i: int) -> tuple[torch.device, torch.Tensor]:
        dev = devices[i]
        a, b = starts[i], starts[i] + sizes[i]
        return dev, traces[a:b].to(dev, non_blocking=True)

    if any(s == 0 for s in sizes):
        # Degenerate split (e.g. n < n_dev): single-device path
        dev = devices[0]
        stl = make_stl(dev)
        t = traces.to(dev)
        return _min_rob0_one_device_try_full_then_chunk(stl, t, chunk_size)

    if n_dev == 2:
        def _work(i: int) -> float:
            dev, shard = _shard(i)
            if dev.type == "cuda":
                torch.cuda.set_device(dev)
            stl = make_stl(dev)
            return _min_rob0_one_device_try_full_then_chunk(stl, shard, chunk_size)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(_work, i) for i in range(2)]
            vals = [f.result() for f in futs]
        return min(vals)

    # 3+ devices: sequential shards (rare)
    vals = []
    for i in range(n_dev):
        dev, shard = _shard(i)
        stl = make_stl(dev)
        vals.append(_min_rob0_one_device_try_full_then_chunk(stl, shard, chunk_size))
    return min(vals)


def max_rob0_vmap(
    make_stl: Callable[[torch.device], torch.nn.Module],
    traces: torch.Tensor,
    primary_device: torch.device,
    eval_devices: Optional[Sequence[torch.device]] = None,
    chunk_size: Optional[int] = None,
) -> float:
    """max_i rho(phi, trace_i) at t=0 (e.g. negated formula in param synth)."""
    devices = _resolve_eval_devices(traces, primary_device, eval_devices)
    n = traces.shape[0]
    if n == 0:
        return -1e9

    if len(devices) == 1:
        dev = devices[0]
        stl = make_stl(dev)
        t = traces.to(dev)
        return _max_rob0_one_device_try_full_then_chunk(stl, t, chunk_size)

    n_dev = len(devices)
    sizes = [n // n_dev + (1 if i < n % n_dev else 0) for i in range(n_dev)]
    starts = [0]
    for s in sizes[:-1]:
        starts.append(starts[-1] + s)

    def _shard(i: int) -> tuple[torch.device, torch.Tensor]:
        dev = devices[i]
        a, b = starts[i], starts[i] + sizes[i]
        return dev, traces[a:b].to(dev, non_blocking=True)

    if any(s == 0 for s in sizes):
        dev = devices[0]
        stl = make_stl(dev)
        t = traces.to(dev)
        return _max_rob0_one_device_try_full_then_chunk(stl, t, chunk_size)

    if n_dev == 2:
        def _work(i: int) -> float:
            dev, shard = _shard(i)
            if dev.type == "cuda":
                torch.cuda.set_device(dev)
            stl = make_stl(dev)
            return _max_rob0_one_device_try_full_then_chunk(stl, shard, chunk_size)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(_work, i) for i in range(2)]
            vals = [f.result() for f in futs]
        return max(vals)

    vals = []
    for i in range(n_dev):
        dev, shard = _shard(i)
        stl = make_stl(dev)
        vals.append(_max_rob0_one_device_try_full_then_chunk(stl, shard, chunk_size))
    return max(vals)


# Backwards-compatible: fixed chunk size (no full-batch attempt first)
def min_rob0_vmap_chunked(
    stl_formula: torch.nn.Module,
    traces: torch.Tensor,
    chunk_size: int = 16,
) -> float:
    dev = traces.device
    return min_rob0_vmap(
        lambda d: stl_formula.to(d),
        traces,
        dev,
        eval_devices=(dev,),
        chunk_size=chunk_size,
    )


def max_rob0_vmap_chunked(
    stl_formula: torch.nn.Module,
    traces: torch.Tensor,
    chunk_size: int = 16,
) -> float:
    dev = traces.device
    return max_rob0_vmap(
        lambda d: stl_formula.to(d),
        traces,
        dev,
        eval_devices=(dev,),
        chunk_size=chunk_size,
    )
