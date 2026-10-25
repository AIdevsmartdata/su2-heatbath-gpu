#!/usr/bin/env python3
"""T1: exact test of the heat-bath sampler for x0 = 1/2 Tr(U V).

Target density p(x) ~ sqrt(1-x^2) exp(alpha x) on [-1, 1]. Exact moments by Gauss-Chebyshev quadrature
of the second kind (weight sqrt(1-x^2), 4000 nodes) compared with the empirical moments produced on the
GPU by `--test-hb ALPHA N` (the same device function as the heat bath). Frozen threshold |z| < 4 for
m1 and m2 at every alpha (standard error = exact standard deviation / sqrt(N)). Both branches are
tested: Creutz (alpha < 2) and Kennedy-Pendleton (alpha >= 2).
Usage: test_heatbath.py BINARY [--out results.json] [--n N]
"""
import argparse
import json
import subprocess
import sys

import numpy as np

ALPHAS = [0.3, 1.0, 1.999, 2.0, 5.0, 20.0, 100.0]


def exact(alpha, n=4000):
    k = np.arange(1, n + 1)
    x = np.cos(k * np.pi / (n + 1))
    w = np.pi / (n + 1) * np.sin(k * np.pi / (n + 1)) ** 2
    f = np.exp(alpha * (x - 1))
    Z = np.sum(w * f)
    return [np.sum(w * f * x ** p) / Z for p in (1, 2, 3, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=20_000_000)
    a = ap.parse_args()
    res = {"n": a.n, "threshold_z": 4.0, "alphas": {}}
    ok_all = True
    for i, alpha in enumerate(ALPHAS):
        out = subprocess.run([a.binary, "--test-hb", repr(alpha), str(a.n), "--seed", str(9000 + i)], capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("TESTHB")]
        if out.returncode != 0 or not line:
            print("FAILED to run", alpha, out.returncode, out.stderr[-500:])
            sys.exit(1)
        kv = dict(t.split("=") for t in line[0].split()[1:])
        m1, m2 = float(kv["m1"]), float(kv["m2"])
        e1, e2, e3, e4 = exact(alpha)
        sd1, sd2 = np.sqrt(e2 - e1 ** 2), np.sqrt(e4 - e2 ** 2)
        z1, z2 = (m1 - e1) / (sd1 / np.sqrt(a.n)), (m2 - e2) / (sd2 / np.sqrt(a.n))
        cap = int(kv["plafond"])
        ok = abs(z1) < 4 and abs(z2) < 4 and cap == 0
        ok_all &= ok
        branch = "Creutz" if alpha < 2 else "Kennedy-Pendleton"
        res["alphas"][repr(alpha)] = dict(branch=branch, m1=m1, m1_exact=e1, z1=z1, m2=m2, m2_exact=e2, z2=z2, cap=cap, ok=bool(ok))
        print(f"alpha={alpha:8.3f} {branch:18s} m1={m1:.8f} exact={e1:.8f} z1={z1:+.2f} m2={m2:.8f} exact={e2:.8f} z2={z2:+.2f} ok={ok}", flush=True)
    res["ok"] = bool(ok_all)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print("TEST_HEATBATH", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
