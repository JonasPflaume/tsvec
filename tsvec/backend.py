"""Runtime backend switches for optional acceleration paths."""

from __future__ import annotations

import os


def use_jaxkd_cuda() -> bool:
    """Return whether jaxkd should use its optional CUDA extension."""
    value = os.environ.get("TSVEC_JAXKD_CUDA", "0").strip().lower()
    return value in {"1", "true", "yes", "on", "cuda", "gpu"}
