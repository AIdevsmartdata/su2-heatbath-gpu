#!/usr/bin/env python3
"""T8: same algorithm on two back-ends. Compares the dumps of two runs made with identical arguments
(e.g. GPU and cpu/su2_omp): max |dU| per dump index. Gate: the initial configurations (cfg_000000.npy)
agree to < 1e-13 (identical Philox counters; only libm rounding differs). Later indices are printed for
information: two back-ends with different libm decorrelate chaotically after some steps.
Usage: compare_configs.py RUN_A RUN_B
"""
import glob
import os
import sys

import numpy as np


def main():
    a, b = sys.argv[1], sys.argv[2]
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a, "cfg_*.npy")))
    if "cfg_000000.npy" not in names:
        print("TEST_SAME_ALGO FAIL (no initial dump; run with --dump-initial)")
        sys.exit(1)
    out = []
    for nm in names:
        pb = os.path.join(b, nm)
        if not os.path.exists(pb):
            continue
        d = float(np.abs(np.load(os.path.join(a, nm)) - np.load(pb)).max())
        out.append((nm, d))
    d0 = dict(out)["cfg_000000.npy"]
    ok = d0 < 1e-13
    print("TEST_SAME_ALGO max|dU| " + " ".join(f"{nm[4:10]}:{d:.1e}" for nm, d in out) + f" initial={d0:.1e} {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
