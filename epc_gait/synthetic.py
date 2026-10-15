"""Synthetic heel-pressure (VGRF-like) signal for demos and tests.

The clinical recordings used in the study are not public, so this module
generates a signal with similar characteristics: one pressure pulse per stride,
sampled at 128 Hz, with slow fluctuations of walking speed, stride-to-stride
amplitude variability, sensor noise and a few abnormal strides.
"""

import numpy as np


def _stride_shape(theta):
    """Pressure profile of one stride as a function of stride phase (0..1)."""
    contact = 1 / (1 + np.exp(-(theta - 0.04) / 0.012))
    release = 1 / (1 + np.exp((theta - 0.52) / 0.015))
    main_peak = 0.62 * np.exp(-((theta - 0.17) / 0.09) ** 2)
    shoulder = 0.22 * np.exp(-((theta - 0.42) / 0.06) ** 2)
    return 0.1 + (main_peak + shoulder + 0.05) * contact * release


def synthetic_vgrf(n_strides=80, fs=128.0, stride_time=1.1, speed_fluctuation=0.06,
                   amplitude_sd=0.05, noise_sd=0.008, n_abnormal=4, seed=0):
    """Generate a synthetic heel-pressure signal (in volts).

    Returns the signal and its sampling frequency.
    """
    rng = np.random.default_rng(seed)
    duration = n_strides * stride_time * 1.15
    t = np.arange(0, duration, 1 / fs)

    # instantaneous stride frequency with slow speed fluctuations
    f0 = 1 / stride_time
    drift = np.cumsum(rng.normal(0, 0.002, len(t)))
    drift -= np.linspace(drift[0], drift[-1], len(t))
    freq = f0 * (1 + speed_fluctuation * np.sin(2 * np.pi * t / 25) + drift)
    stride_pos = np.cumsum(freq) / fs  # stride number at each sample
    keep = stride_pos < n_strides
    t, stride_pos = t[keep], stride_pos[keep]

    stride_idx = stride_pos.astype(int)
    theta = stride_pos - stride_idx
    amplitude = 1 + rng.normal(0, amplitude_sd, n_strides)
    signal = _stride_shape(theta) * amplitude[stride_idx]

    # a few abnormal strides (e.g., a stumble or a sensor glitch)
    for s in rng.choice(np.arange(5, n_strides - 5), n_abnormal, replace=False):
        m = stride_idx == s
        signal[m] = 0.1 + 0.8 * (signal[m] - 0.1) + 0.12 * np.sin(6 * np.pi * theta[m])

    signal += rng.normal(0, noise_sd, len(signal))
    return signal, fs
