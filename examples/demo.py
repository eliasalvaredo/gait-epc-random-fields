"""End-to-end demo of the EPC pipeline on a synthetic heel-pressure signal.

Run from the repository root:

    python examples/demo.py

Figures are saved to docs/figures/.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from epc_gait import (angular_resample, build_random_field, epc_curve,  # noqa: E402
                      excursion_set, fit_dmepc, dmepc, polynomize,
                      remove_high_sd_cycles, split_params, synthetic_vgrf)

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
SPS = 200            # samples per stride after angular resampling
U = np.arange(-0.2, 1.8, 0.01)  # excursion set levels (V)
SM = 4               # number of Gaussian components in the DMEPC model
NRMSE_THRESHOLD = 0.4


def main(seed=0):
    os.makedirs(OUT, exist_ok=True)

    # 1. Signal
    x, fs = synthetic_vgrf(seed=seed)
    print(f"Synthetic signal: {len(x)} samples at {fs:.0f} Hz ({len(x) / fs:.0f} s)")

    # 2. Angular resampling
    x_res, _, info = angular_resample(x, samples_per_stride=SPS)
    period = int(round(info["period"]))
    print(f"Mean stride period: {info['period']:.1f} samples; "
          f"{len(x_res) // SPS} strides after resampling")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for p in info["peaks"][1:-1]:
        axes[0].plot(x[p - period // 2:p + period // 2], lw=0.6)
    for k in range(1, len(x_res) // SPS - 1):
        axes[1].plot(x_res[k * SPS:(k + 1) * SPS], lw=0.6)
    axes[0].set(title="Original signal: overlapped strides", xlabel="Sample",
                ylabel="Pressure (V)")
    axes[1].set(title="After angular resampling (200 samples/stride)",
                xlabel="Stride position (samples)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_angular_resampling.png"), dpi=110)

    # 3. Random field: cycles x samples, cleaned and smoothed
    rf = build_random_field(x_res, samples_per_stride=SPS)
    rf_clean, removed = remove_high_sd_cycles(rf, fraction=0.15)
    rf_poly, degree = polynomize(rf_clean)
    print(f"Random field: {rf.shape[0]} cycles x {rf.shape[1]} samples; "
          f"{len(removed)} high-deviation cycles removed; polynomial degree {degree}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, field, title in zip(axes, [rf, rf_clean, rf_poly],
                                ["All cycles", "High-deviation cycles removed",
                                 f"Polynomial smoothing (degree {degree})"]):
        for cycle in field:
            ax.plot(cycle, lw=0.5)
        ax.set(title=title, xlabel="Sample within cycle")
    axes[0].set_ylabel("Pressure (V)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_cycle_cleaning.png"), dpi=110)

    # 4. Excursion sets and EPC
    epc = epc_curve(rf_poly, U)
    u_show = 0.6
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    im = axes[0].imshow(rf_poly.T, aspect="auto", origin="lower", cmap="plasma")
    fig.colorbar(im, ax=axes[0], label="Pressure (V)")
    axes[0].set(title="Random field", xlabel="Cycle", ylabel="Sample within cycle")
    axes[1].imshow(excursion_set(rf_poly, u_show).T, aspect="auto", origin="lower",
                   cmap="gray")
    axes[1].set(title=f"Excursion set at u = {u_show} V", xlabel="Cycle")
    axes[2].plot(U, epc, color="tab:red")
    axes[2].axvline(u_show, color="gray", ls="--", lw=0.8)
    axes[2].set(title="EPC vs excursion level", xlabel="u (V)", ylabel="EPC")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "03_excursion_sets_epc.png"), dpi=110)

    # 5. DMEPC model fitted with PSO
    params, cost = fit_dmepc(U, epc, sm=SM, seed=seed)
    status = "accepted" if cost < NRMSE_THRESHOLD else "above the 0.4 acceptance threshold"
    print(f"DMEPC fit (Sm={SM}): NRMSE = {cost:.3f} ({status})")
    for name, values in split_params(params, SM).items():
        print(f"  {name:>5}: " + "  ".join(f"{v:9.3f}" for v in values))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(U, epc, label="Measured EPC")
    ax.plot(U, dmepc(U, params, SM), label=f"DMEPC (Sm={SM}), NRMSE={cost:.2f}")
    ax.set(xlabel="u (V)", ylabel="EPC", title="DMEPC model fitted with PSO")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "04_dmepc_fit.png"), dpi=110)
    print(f"Figures saved to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
