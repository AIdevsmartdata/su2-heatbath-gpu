"""Shared statistics helpers for the tests (numpy only)."""
import csv
import os
from math import exp, lgamma, log

import numpy as np


def tau_int(x, c=6.0):
    """Integrated autocorrelation time with Sokal's automatic window: tau = 1/2 + sum_{t=1}^{W} rho(t),
    W = first t with t >= c * tau(t). Returns (tau, W)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    v = x @ x / n
    if v == 0 or n < 8:
        return 0.5, 0
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n] / np.arange(n, 0, -1) / v
    t, W = 0.5, 0
    for w in range(1, n // 4):
        t += ac[w]
        W = w
        if w >= c * t:
            break
    return float(max(t, 0.5)), int(W)


def mean_err(x):
    """Mean and error sqrt(2 tau var / n)."""
    x = np.asarray(x, float)
    t, _ = tau_int(x)
    return float(x.mean()), float(np.sqrt(2 * t * x.var(ddof=1) / len(x))), t


def read_plaquettes(run_dir):
    """'sample' plaquettes of measurements.csv written by su2_hip / su2_cuda / su2_omp."""
    with open(os.path.join(run_dir, "measurements.csv")) as f:
        return np.array([float(r["plaquette"]) for r in csv.DictReader(f) if r["phase"] == "sample"])


def read_reference_csv(path):
    with open(path) as f:
        return np.array([float(r["plaquette"]) for r in csv.DictReader(f)])


def bessel_i(nu, x, terms=80):
    """Modified Bessel function I_nu(x), integer nu >= 0, by its power series (fine for x <= ~10)."""
    return sum(exp((2 * k + nu) * log(x / 2) - lgamma(k + 1) - lgamma(k + nu + 1)) for k in range(terms))


def strong_coupling(beta):
    """Leading strong-coupling value of <1/2 Tr U_p> in 4D SU(2): u + 4 u^5, u = I2/I1,
    with the conservative bound 50 u^7 on the neglected terms (see docs/VALIDATION.md)."""
    u = bessel_i(2, beta) / bessel_i(1, beta)
    return u + 4 * u ** 5, 50 * u ** 7, u
