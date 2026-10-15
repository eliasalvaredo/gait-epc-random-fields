"""Topological analysis of gait signals with the Euler-Poincare Characteristic."""

from .resampling import angular_resample, detect_stride_peaks
from .random_field import build_random_field, remove_high_sd_cycles, polynomize
from .epc import excursion_set, euler_characteristic, epc_curve
from .dmepc import dmepc, fit_dmepc, split_params
from .synthetic import synthetic_vgrf

__all__ = [
    "angular_resample", "detect_stride_peaks",
    "build_random_field", "remove_high_sd_cycles", "polynomize",
    "excursion_set", "euler_characteristic", "epc_curve",
    "dmepc", "fit_dmepc", "split_params",
    "synthetic_vgrf",
]
