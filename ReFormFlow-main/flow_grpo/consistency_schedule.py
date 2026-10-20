"""Consistency coefficients deterministic conditional on the observation noise.

This is a regularization choice, not an improvement in posterior estimation.
Avoid ESS-dependent mixing: ESS is random conditional on y and can correlate
with the posterior estimate, changing the expected update.
"""
import math


def consistency_weight(sigma, amplitude, schedule="constant"):
    sigma, amplitude = float(sigma), float(amplitude)
    if not math.isfinite(sigma) or not 0 <= sigma <= 1:
        raise ValueError("sigma must be in [0,1]")
    if not math.isfinite(amplitude) or not 0 <= amplitude <= 1:
        raise ValueError("amplitude must be in [0,1]")
    if schedule == "constant":
        return amplitude
    if schedule == "linear_alpha":
        return amplitude * (1 - sigma)
    raise ValueError("Unknown deterministic consistency schedule: " + schedule)
