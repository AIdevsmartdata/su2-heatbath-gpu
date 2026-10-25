#!/usr/bin/env python3
"""Plaquette tests T2/T3/T4/C1/C2 (criteria frozen in tests/CRITERIA.md).

  plaquette_tests.py strong  NAME SOURCE --beta B            # |P - (u + 4u^5)| < 4 sigma_tot
  plaquette_tests.py regress NAME SOURCE --value V --err E   # |z| < 3 against an archived value
  plaquette_tests.py compare NAME SOURCE_A SOURCE_B          # |z| < 3 between two Monte Carlo estimates
SOURCE is a run directory (measurements.csv of su2_hip / su2_cuda / su2_omp) or a CSV file written by
reference/su2_metropolis_ref (column 'plaquette'). Prints one line 'TEST_<NAME> ... PASS|FAIL'.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import mean_err, read_plaquettes, read_reference_csv, strong_coupling  # noqa: E402


def load(src):
    x = read_plaquettes(src) if os.path.isdir(src) else read_reference_csv(src)
    m, e, t = mean_err(x)
    return m, e, t, len(x)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("strong"); s.add_argument("name"); s.add_argument("src"); s.add_argument("--beta", type=float, required=True)
    r = sp.add_parser("regress"); r.add_argument("name"); r.add_argument("src")
    r.add_argument("--value", type=float, required=True); r.add_argument("--err", type=float, required=True)
    c = sp.add_parser("compare"); c.add_argument("name"); c.add_argument("a"); c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "strong":
        m, e, t, n = load(a.src)
        exp_, th, u = strong_coupling(a.beta)
        tot = float(np.hypot(e, th))
        z = (m - exp_) / tot
        z_lead = (m - u) / tot
        ok = abs(z) < 4
        print(f"TEST_{a.name} beta={a.beta} P={m:.7f}({e:.1e}) tau={t:.2f} n={n} expected(u+4u^5)={exp_:.7f} sigma_th={th:.1e} "
              f"z={z:+.2f} [info: z against u alone = {z_lead:+.2f}] {'PASS' if ok else 'FAIL'}")
    elif a.cmd == "regress":
        m, e, t, n = load(a.src)
        z = (m - a.value) / float(np.hypot(e, a.err))
        ok = abs(z) < 3
        print(f"TEST_{a.name} P={m:.7f}({e:.1e}) tau={t:.2f} n={n} archived={a.value:.7f}({a.err:.1e}) z={z:+.2f} {'PASS' if ok else 'FAIL'}")
    else:
        ma, ea, ta, na = load(a.a)
        mb, eb, tb, nb = load(a.b)
        z = (ma - mb) / float(np.hypot(ea, eb))
        ok = abs(z) < 3
        print(f"TEST_{a.name} A={ma:.7f}({ea:.1e}, tau={ta:.2f}, n={na}) B={mb:.7f}({eb:.1e}, tau={tb:.2f}, n={nb}) z={z:+.2f} "
              f"{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
