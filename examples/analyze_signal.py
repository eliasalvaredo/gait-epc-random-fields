"""Run the EPC pipeline on your own pressure signal.

The input is a plain-text file with one sample per row (several columns are
allowed; choose one with --column), for example the VGRF recordings used in the
study, which were sampled at 128 Hz.

    python examples/analyze_signal.py my_signal.txt --column 0 --u-max 1.8

Outputs a CSV file with the EPC curve and the fitted DMEPC parameters.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from epc_gait import (angular_resample, build_random_field, epc_curve,  # noqa: E402
                      fit_dmepc, polynomize, remove_high_sd_cycles, split_params)

MIN_CYCLES = 30  # signals with fewer cycles were excluded in the study


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("signal", help="text file with the pressure signal")
    parser.add_argument("--column", type=int, default=0, help="column to use (default 0)")
    parser.add_argument("--u-max", type=float, default=None,
                        help="maximum excursion level in volts (default: signal maximum)")
    parser.add_argument("--sm", type=int, default=4, help="DMEPC components (default 4)")
    parser.add_argument("--no-poly", action="store_true", help="skip polynomial smoothing")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.loadtxt(args.signal, ndmin=2)[:, args.column]
    x_res, _, info = angular_resample(data)
    rf = build_random_field(x_res)
    if rf.shape[0] < MIN_CYCLES:
        print(f"Warning: only {rf.shape[0]} cycles (the study required {MIN_CYCLES}).")
    rf, _ = remove_high_sd_cycles(rf)
    if not args.no_poly:
        rf, degree = polynomize(rf)
        print(f"Polynomial degree: {degree}")

    u_max = args.u_max if args.u_max is not None else np.ceil(rf.max() * 10) / 10
    u = np.arange(-0.2, u_max, 0.01)
    epc = epc_curve(rf, u)
    params, cost = fit_dmepc(u, epc, sm=args.sm, seed=args.seed)

    base = os.path.splitext(args.signal)[0]
    np.savetxt(base + "_EPC.csv", np.column_stack([u, epc]), delimiter=",",
               header="u,epc", comments="")
    np.savetxt(base + "_DMEPC_params.csv", params[None, :], delimiter=",",
               header=",".join(f"{n}{k + 1}" for n in ["A", "B", "sigma", "mu"]
                               for k in range(args.sm)), comments="")
    print(f"Random field: {rf.shape}; DMEPC NRMSE = {cost:.3f}")
    for name, values in split_params(params, args.sm).items():
        print(f"  {name:>5}: " + "  ".join(f"{v:9.3f}" for v in values))
    print(f"Saved {base}_EPC.csv and {base}_DMEPC_params.csv")


if __name__ == "__main__":
    main()
