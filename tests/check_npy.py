#!/usr/bin/env python3
"""T6: integrity of the NPY dumps of a run directory (layout, unitarity, plaquette).

For every cfg_XXXXXX.npy: dtype complex128, shape (4, L, L, L, L, 2, 2), C order; SU(2) structure
U11 = conj(U00), U10 = -conj(U01); max |U U^+ - 1|; max |det U - 1|; plaquette recomputed with numpy
(U_mu(x) U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+) compared with the value written in measurements.csv at
the same index. Thresholds: 1e-13 (unitarity, det, structure), 1e-12 (plaquette).
Usage: check_npy.py RUN_DIR [RUN_DIR ...]
"""
import csv
import glob
import os
import sys

import numpy as np


def plaq_numpy(cfg):
    tot = 0.0
    for mu in range(4):
        for nu in range(mu + 1, 4):
            Umu, Unu = cfg[mu], cfg[nu]
            P = Umu @ np.roll(Unu, -1, axis=mu) @ np.conj(np.swapaxes(np.roll(Umu, -1, axis=nu), -1, -2)) @ np.conj(np.swapaxes(Unu, -1, -2))
            tot += 0.5 * np.trace(P, axis1=-2, axis2=-1).real.mean()
    return tot / 6


def main():
    worst = dict(unit=0.0, det=0.0, su2=0.0, plaq=0.0)
    n, shape_ok = 0, True
    for d in sys.argv[1:]:
        meas = {}
        with open(os.path.join(d, "measurements.csv")) as f:
            for r in csv.DictReader(f):
                meas[int(r["measurement_index"])] = float(r["plaquette"])
        therm0 = None
        with open(os.path.join(d, "therm.csv")) as f:
            for r in csv.DictReader(f):
                if int(r["step"]) == 0:
                    therm0 = float(r["plaquette"])
        for c in sorted(glob.glob(os.path.join(d, "cfg_*.npy"))):
            m = int(os.path.basename(c)[4:10])
            cfg = np.load(c)
            L = cfg.shape[1]
            shape_ok &= bool(cfg.dtype == np.complex128 and cfg.shape == (4, L, L, L, L, 2, 2) and cfg.flags["C_CONTIGUOUS"])
            uu = cfg @ np.conj(np.swapaxes(cfg, -1, -2))
            worst["unit"] = max(worst["unit"], float(np.abs(uu - np.eye(2)).max()))
            det = cfg[..., 0, 0] * cfg[..., 1, 1] - cfg[..., 0, 1] * cfg[..., 1, 0]
            worst["det"] = max(worst["det"], float(np.abs(det - 1).max()))
            worst["su2"] = max(worst["su2"], float(max(np.abs(cfg[..., 1, 1] - np.conj(cfg[..., 0, 0])).max(),
                                                       np.abs(cfg[..., 1, 0] + np.conj(cfg[..., 0, 1])).max())))
            # index 0 of a dump is the initial state (therm.csv step 0), other indices are measurement indices
            ref = therm0 if m == 0 else meas.get(m)
            if ref is not None:
                worst["plaq"] = max(worst["plaq"], abs(plaq_numpy(cfg) - ref))
            n += 1
    ok = n > 0 and shape_ok and worst["unit"] < 1e-13 and worst["det"] < 1e-13 and worst["su2"] < 1e-13 and worst["plaq"] < 1e-12
    print(f"CHECK_NPY n_cfg={n} shape_ok={shape_ok} max_unitarity={worst['unit']:.2e} max_det={worst['det']:.2e} "
          f"max_su2_structure={worst['su2']:.2e} max_plaquette_numpy_vs_csv={worst['plaq']:.2e} {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
