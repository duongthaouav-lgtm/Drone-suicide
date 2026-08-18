import gc
import numpy as np
import torch

import config as cfg

PERF_STATS: dict[str, list[float]] = {
    'inference_times': [],
    'preprocess_times': [],
    'postprocess_times': [],
    'total_times': [],
    'memory_usage': [],
}

_MAX_SAMPLES = 100


def reset():
    """Clear all accumulated performance samples."""
    for key in PERF_STATS:
        PERF_STATS[key].clear()


def update(inference_time: float, preprocess_time: float = 0.0, postprocess_time: float = 0.0):
    """Record timing for one processed frame."""
    if not cfg.PERFORMANCE_MONITORING:
        return

    total = inference_time + preprocess_time + postprocess_time
    PERF_STATS['inference_times'].append(inference_time)
    PERF_STATS['preprocess_times'].append(preprocess_time)
    PERF_STATS['postprocess_times'].append(postprocess_time)
    PERF_STATS['total_times'].append(total)

    for key in PERF_STATS:
        if len(PERF_STATS[key]) > _MAX_SAMPLES:
            PERF_STATS[key].pop(0)


def get_summary() -> dict | None:
    """Return a summary dict of average/min/max inference metrics."""
    times = PERF_STATS['inference_times']
    totals = PERF_STATS['total_times']

    if not cfg.PERFORMANCE_MONITORING or not times:
        return None

    avg_total = float(np.mean(totals))
    return {
        'avg_inference_ms': float(np.mean(times)) * 1000,
        'avg_total_ms': avg_total * 1000,
        'min_inference_ms': float(np.min(times)) * 1000,
        'max_inference_ms': float(np.max(times)) * 1000,
        'fps': 1.0 / avg_total if avg_total > 0 else 0.0,
    }


def cleanup_memory():
    """Release unused memory (especially useful on Pi 5 with limited RAM)."""
    if cfg.ENABLE_MEMORY_OPTIMIZATION:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()