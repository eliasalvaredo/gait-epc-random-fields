"""Peak detection and angular resampling of cyclic gait signals.

Angular resampling turns a pressure-vs-time signal into a pressure-vs-stride-position
signal, so that every stride has the same number of samples regardless of
fluctuations in walking speed:

1. Detect the stride peaks and estimate the mean stride period.
2. Isolate the fundamental frequency with a band-pass filter in the Fourier domain
   (keeping positive frequencies only, which gives the analytic signal).
3. Extract the unwrapped phase of the analytic signal: phase / 2*pi is the
   (fractional) stride number at each time sample.
4. Interpolate the signal (cubic splines) on a regular grid of stride positions.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter


def estimate_period(x):
    """Estimate the dominant period (in samples) from the autocorrelation."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    spectrum = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(np.abs(spectrum) ** 2)[:n]
    acf /= acf[0]
    # first local maximum after the autocorrelation drops below zero
    below = np.where(acf < 0)[0]
    start = below[0] if len(below) else 1
    peaks, _ = find_peaks(acf[start:])
    if len(peaks) == 0:
        raise ValueError("Could not estimate the signal period.")
    best = peaks[np.argmax(acf[start:][peaks])]
    return float(start + best)


def detect_stride_peaks(x, period=None, smooth_window=None, polyorder=3):
    """Detect one maximum per stride.

    The signal is smoothed with a Savitzky-Golay filter, and peaks are searched
    with a minimum distance of 70% of the stride period.

    Returns the indices of the maxima and the smoothed signal.
    """
    x = np.asarray(x, float)
    if period is None:
        period = estimate_period(x)
    if smooth_window is None:
        smooth_window = max(polyorder + 2, int(period * 0.1) | 1)  # odd length
    smooth = savgol_filter(x, smooth_window, polyorder)
    prominence = 0.3 * (np.percentile(smooth, 95) - np.percentile(smooth, 5))
    peaks, _ = find_peaks(smooth, distance=int(0.7 * period), prominence=prominence)
    return peaks, smooth


def analytic_phase(x, band):
    """Unwrapped phase of the analytic signal of ``x`` band-passed to ``band``.

    ``band`` is given in normalized frequency (cycles per sample), e.g.
    ``[0.5 * f0, 1.5 * f0]`` around the fundamental frequency ``f0``.
    """
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    spectrum = np.fft.fft(x)
    freqs = np.fft.fftfreq(n)
    mask = (freqs >= band[0]) & (freqs <= band[1])  # positive band only -> analytic
    analytic = np.fft.ifft(spectrum * mask)
    return np.unwrap(np.angle(analytic))


def angular_resample(x, samples_per_stride=200, period=None):
    """Resample ``x`` so that each stride has ``samples_per_stride`` samples.

    Returns
    -------
    signal_res : ndarray
        Signal as a function of the regular stride position.
    stride_res : ndarray
        Stride position of each resampled sample (in strides).
    info : dict
        Intermediate results: period (samples), fundamental frequency,
        stride position of each original sample.
    """
    x = np.asarray(x, float)
    peaks, _ = detect_stride_peaks(x, period=period)
    if len(peaks) < 3:
        raise ValueError("At least 3 strides are needed for angular resampling.")
    per = float(np.mean(np.diff(peaks)))
    f0 = 1.0 / per
    phase = analytic_phase(x, [0.5 * f0, 1.5 * f0])

    stride_t = (phase - phase[0]) / (2 * np.pi)
    # guard against tiny non-monotonic segments caused by noise
    stride_t = np.maximum.accumulate(stride_t)
    stride_t += np.arange(len(stride_t)) * 1e-12

    interpolator = interp1d(stride_t, x, kind="cubic")
    stride_res = np.arange(0, int(stride_t[-1]), 1.0 / samples_per_stride)
    signal_res = interpolator(stride_res)
    info = {"period": per, "f0": f0, "stride_t": stride_t, "peaks": peaks}
    return signal_res, stride_res, info
