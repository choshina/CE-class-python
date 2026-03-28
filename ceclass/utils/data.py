from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


def load_traces(
    source: Union[str, Path, np.ndarray, torch.Tensor],
    signal_indices: Optional[list[int]] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Load falsifying traces from various sources.

    Args:
        source: One of:
            - Path to .mat file (loads via scipy.io.loadmat)
            - Path to .npy file (loads via np.load)
            - numpy array of shape (num_traces, timesteps, dims)
            - torch tensor of shape (num_traces, timesteps, dims)
        signal_indices: If provided, select only these signal columns.
        device: Target torch device.
        dtype: Target torch dtype.

    Returns:
        Tensor of shape (num_traces, timesteps, dims).
    """
    if isinstance(source, torch.Tensor):
        traces = source
    elif isinstance(source, np.ndarray):
        traces = torch.from_numpy(source)
    else:
        path = Path(source)
        if path.suffix == '.mat':
            traces = _load_mat(path)
        elif path.suffix == '.npy':
            traces = torch.from_numpy(np.load(path))
        elif path.suffix == '.npz':
            data = np.load(path)
            key = list(data.keys())[0]
            traces = torch.from_numpy(data[key])
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    traces = traces.to(dtype=dtype)

    if signal_indices is not None:
        traces = traces[:, :, signal_indices]

    if device is not None:
        traces = traces.to(device)

    if traces.ndim == 2:
        # Assume (num_traces, timesteps) — add a signal dimension at the end.
        traces = traces.unsqueeze(-1)

    return traces


def _load_mat(path: Path) -> torch.Tensor:
    """Load traces from a MATLAB .mat file."""
    try:
        import scipy.io as sio
    except ImportError:
        raise ImportError("scipy required for loading .mat files: pip install scipy")

    mat = sio.loadmat(str(path))

    # Try common variable names
    for key in ['traces', 'data', 'signals', 'X']:
        if key in mat:
            return torch.from_numpy(np.array(mat[key], dtype=np.float64))

    # Fall back to first non-metadata key
    data_keys = [k for k in mat.keys() if not k.startswith('__')]
    if data_keys:
        return torch.from_numpy(np.array(mat[data_keys[0]], dtype=np.float64))

    raise ValueError(f"No trace data found in {path}. Keys: {list(mat.keys())}")
