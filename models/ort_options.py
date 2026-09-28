"""ONNX Runtime session options shared by the ONNX-backed models."""

from __future__ import annotations

from typing import Any


def session_options(intra_op_threads: int = 0) -> Any:
    """SessionOptions with a capped intra-op thread pool (0 = runtime default).

    Fire and face presence inference run at the same time in
    ACTIVE_DETECTION; by default each session sizes its pool to the CPU's
    performance cores, so the two sessions fight over the same cores.
    """
    import onnxruntime as ort  # deferred: importing onnxruntime is slow

    options = ort.SessionOptions()
    if intra_op_threads > 0:
        options.intra_op_num_threads = intra_op_threads
    return options
