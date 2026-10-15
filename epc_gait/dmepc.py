"""Decomposed Mean EPC (DMEPC) model fitted with Particle Swarm Optimization.

A non-Gaussian random field is modelled as a combination of ``Sm`` signed Gaussian
random fields (Ramos-Martinez, Corbier et al., IEEE Access, 2021). Its mean EPC is
approximated by

    DMEPC(u) = sum_k s_k * [ A_k * exp(-X_k(u)^2 / 2) + B_k * Phi(X_k(u)) ],
    X_k(u) = (u - mu_k) / sigma_k,

with alternating signs s_k = -1, +1, -1, ... and Phi the standard normal CDF, as in
the original implementation. The 4 * Sm parameters are estimated by minimizing the
normalized RMSE (NRMSE) between the measured EPC and the model.

Parameter vector layout: [A_1..A_Sm, B_1..B_Sm, sigma_1..sigma_Sm, mu_1..mu_Sm].
"""

import numpy as np
from scipy.stats import norm


def dmepc(u, params, sm):
    """Evaluate the DMEPC model for one or several parameter vectors.

    ``params`` can have shape (4*sm,) or (n_particles, 4*sm).
    """
    params = np.atleast_2d(params)
    a, b, sigma, mu = (params[:, i * sm:(i + 1) * sm] for i in range(4))
    signs = np.where(np.arange(1, sm + 1) % 2 == 0, 1.0, -1.0)
    x = (u[None, None, :] - mu[:, :, None]) / sigma[:, :, None]
    waves = a[:, :, None] * np.exp(-x ** 2 / 2) + b[:, :, None] * norm.cdf(x)
    out = np.einsum("k,pku->pu", signs, waves)
    return out[0] if out.shape[0] == 1 else out


def nrmse_batch(estimated, target):
    """NRMSE of each row of ``estimated`` with respect to ``target``."""
    rmse = np.sqrt(np.mean((estimated - target) ** 2, axis=-1))
    return rmse / np.std(target)


def default_bounds(sm, u):
    """Search bounds: A, B in [-1e5, 1e5]; sigma in [du, 1]; mu within the u range.

    The original code allowed sigma in [-1, 1]. Here sigma is kept positive and at
    least as wide as the spacing ``du`` of the u grid: narrower components would
    fall between grid points and only produce degenerate solutions.
    """
    du = float(np.min(np.diff(u))) if len(u) > 1 else 1e-3
    low = np.concatenate([np.full(sm, -1e5), np.full(sm, -1e5),
                          np.full(sm, du), np.full(sm, u.min())])
    high = np.concatenate([np.full(sm, 1e5), np.full(sm, 1e5),
                           np.full(sm, 1.0), np.full(sm, u.max())])
    return low, high


def fit_dmepc(u, epc, sm=4, n_particles=25, iters=2000, w=0.3, c1=1.2, c2=2.7,
              n_restarts=5, bounds=None, seed=None):
    """Fit the DMEPC model to a measured EPC curve with global-best PSO.

    The PSO hyperparameters (25 particles, w=0.3, c1=1.2, c2=2.7, 2000 iterations)
    are the ones used in the original study. The swarm is restarted
    ``n_restarts`` times and the best solution is kept.

    Returns the best parameter vector and its NRMSE.
    """
    rng = np.random.default_rng(seed)
    u = np.asarray(u, float)
    epc = np.asarray(epc, float)
    bounds = bounds if bounds is not None else default_bounds(sm, u)
    best_params, best_cost = None, np.inf
    for _ in range(n_restarts):
        params, cost = _pso(u, epc, sm, n_particles, iters, w, c1, c2, bounds, rng)
        if cost < best_cost:
            best_params, best_cost = params, cost
    return best_params, float(best_cost)


def _pso(u, epc, sm, n_particles, iters, w, c1, c2, bounds, rng):
    """One run of global-best PSO."""
    low, high = bounds
    dim = 4 * sm

    # Initialization: A and B on the scale of the EPC values, sigma small, and mu
    # near the levels where the EPC departs most from its baseline.
    scale = max(1.0, np.abs(epc).max())
    init_low, init_high = low.copy(), high.copy()
    init_low[:2 * sm], init_high[:2 * sm] = -scale, scale
    init_high[2 * sm:3 * sm] = min(high[2 * sm], 0.3)
    pos = rng.uniform(init_low, init_high, size=(n_particles, dim))
    weights = np.abs(epc - np.median(epc)) + 1e-9
    pos[:, 3 * sm:] = rng.choice(u, size=(n_particles, sm), p=weights / weights.sum())
    vel = rng.uniform(-1, 1, size=(n_particles, dim)) * (init_high - init_low) * 0.1

    def cost(p):
        with np.errstate(all="ignore"):
            c = nrmse_batch(dmepc(u, p, sm).reshape(len(p), -1), epc)
        return np.where(np.isfinite(c), c, np.inf)

    personal_best = pos.copy()
    personal_cost = cost(pos)
    g = np.argmin(personal_cost)
    global_best, global_cost = personal_best[g].copy(), personal_cost[g]

    for _ in range(iters):
        r1, r2 = rng.random((2, n_particles, dim))
        vel = (w * vel + c1 * r1 * (personal_best - pos)
               + c2 * r2 * (global_best - pos))
        pos = np.clip(pos + vel, low, high)
        current = cost(pos)
        improved = current < personal_cost
        personal_best[improved] = pos[improved]
        personal_cost[improved] = current[improved]
        g = np.argmin(personal_cost)
        if personal_cost[g] < global_cost:
            global_best, global_cost = personal_best[g].copy(), personal_cost[g]

    return global_best, global_cost


def split_params(params, sm):
    """Return the parameters as a dict of arrays: A, B, sigma, mu."""
    params = np.asarray(params)
    return {name: params[i * sm:(i + 1) * sm]
            for i, name in enumerate(["A", "B", "sigma", "mu"])}
