"""Excursion sets and Euler-Poincare Characteristic (EPC) of 2D random fields."""

import numpy as np


def excursion_set(rf, u):
    """Binary image of the points where the random field reaches level ``u``."""
    return (np.asarray(rf) >= u).astype(np.uint8)


def euler_characteristic(image):
    """Euler characteristic of a binary image, with 8-connectivity.

    Uses the local 2x2 pattern ("quad bit") counting method of Gray (1971):

        E = (C1 - C3 - 2 * Cd) / 4

    where C1, C3 and Cd are the number of 2x2 windows with exactly one foreground
    pixel, exactly three foreground pixels, and two diagonal foreground pixels.
    The result equals the number of connected components minus the number of holes.
    """
    img = np.pad(np.asarray(image, dtype=np.int8) != 0, 1).astype(np.int8)
    a = img[:-1, :-1]
    b = img[:-1, 1:]
    c = img[1:, :-1]
    d = img[1:, 1:]
    total = a + b + c + d
    c1 = np.count_nonzero(total == 1)
    c3 = np.count_nonzero(total == 3)
    diagonal = (total == 2) & (a == d)
    cd = np.count_nonzero(diagonal)
    return (c1 - c3 - 2 * cd) // 4


def epc_curve(rf, u_values):
    """EPC of the excursion sets of ``rf`` for every level in ``u_values``."""
    return np.array([euler_characteristic(excursion_set(rf, u)) for u in u_values],
                    dtype=float)
