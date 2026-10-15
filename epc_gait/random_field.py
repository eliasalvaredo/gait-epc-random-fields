"""Build a 2D random field (cycles x samples) from an angularly resampled signal."""

from collections import Counter

import numpy as np
from numpy.polynomial import Polynomial

from .resampling import detect_stride_peaks


def build_random_field(signal_res, samples_per_stride=200, before=70, after=100,
                       skip_first=2):
    """Stack the stride cycles of a resampled signal into a 2D matrix.

    Each row is one cycle: ``before`` samples before its maximum and ``after``
    samples after it (170 samples by default). The first ``skip_first`` cycles are
    discarded because their shape usually differs from the rest, and incomplete
    cycles at the edges are dropped.
    """
    peaks, _ = detect_stride_peaks(signal_res, period=samples_per_stride)
    peaks = peaks[skip_first:]
    cycles = [signal_res[p - before:p + after] for p in peaks
              if p - before >= 0 and p + after <= len(signal_res)]
    if not cycles:
        raise ValueError("No complete cycles found.")
    return np.vstack(cycles)


def remove_high_sd_cycles(rf, fraction=0.15):
    """Remove the ``fraction`` of cycles that deviate most from the mean cycle.

    The deviation of each cycle is the root mean square of its difference to the
    average cycle. Returns the cleaned field and the indices of removed cycles.
    """
    mean_cycle = rf.mean(axis=0)
    sd = np.sqrt(np.mean((rf - mean_cycle) ** 2, axis=1))
    n_remove = int(fraction * rf.shape[0])
    removed = np.argsort(sd)[::-1][:n_remove]
    return np.delete(rf, removed, axis=0), np.sort(removed)


def nrmse(estimated, target):
    """Root mean square error normalized by the standard deviation of the target."""
    target = np.asarray(target, float)
    rmse = np.sqrt(np.mean((target - estimated) ** 2))
    return rmse / np.sqrt(np.mean((target - target.mean()) ** 2))


def polynomize(rf, max_degree=11):
    """Smooth each cycle with a polynomial (Random Algebraic Polynomials).

    The EPC is very sensitive to small fluctuations, so each cycle is replaced by a
    polynomial fit. For every cycle, the degree (0..``max_degree``) with the lowest
    NRMSE is found; the most frequent of these degrees is then used for all cycles.

    Returns the smoothed field and the selected degree.
    """
    x = np.arange(rf.shape[1])
    best_degrees = []
    for cycle in rf:
        errors = [nrmse(Polynomial.fit(x, cycle, d)(x), cycle)
                  for d in range(max_degree + 1)]
        best_degrees.append(int(np.argmin(errors)))
    degree = Counter(best_degrees).most_common(1)[0][0]
    smoothed = np.vstack([Polynomial.fit(x, cycle, degree)(x) for cycle in rf])
    return smoothed, degree
