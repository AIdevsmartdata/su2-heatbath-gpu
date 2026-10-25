#!/usr/bin/env python3
"""T7 / run sentinel: post-hoc consistency check of su2_hip / su2_cuda / su2_omp output directories.

Why: on one integrated AMD GPU (gfx1151) we observed runs that ended with exit code 0 and a COMPLETE
marker although the trajectory contained NaN (see README, "Silent data corruption"). The generator
itself has no in-loop sentinel, so check every run you rely on.
Checks: COMPLETE file; manifest.json parses and its status ("statut") is COMPLETE; every value of
therm.csv and measurements.csv is finite; |plaquette| <= 1 and |P_mu| <= 1; indices contiguous;
optional log file: the PERF line reports a heat-bath trial cap of 0 ("plafond_atteint=0" / "caps=0")
and a mean number of trials per link between 1 and 2.
Usage: check_run.py RUN_DIR [--log LOGFILE] [RUN_DIR [--log LOGFILE] ...]
Exit code: number of failing directories.
"""
import csv
import json
import math
import os
import re
import sys


def check(d, log=None):
    errs = []
    if not os.path.exists(os.path.join(d, "COMPLETE")):
        errs.append("no COMPLETE marker")
    try:
        man = json.load(open(os.path.join(d, "manifest.json")))
        if man.get("statut") != "COMPLETE":
            errs.append(f"manifest status = {man.get('statut')!r}")
    except Exception as e:  # noqa: BLE001
        errs.append(f"manifest unreadable: {e}")
    for name, idx in (("therm.csv", "step"), ("measurements.csv", "measurement_index")):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            errs.append(f"missing {name}")
            continue
        rows = list(csv.DictReader(open(p)))
        if [int(r[idx]) for r in rows] != list(range(len(rows))):
            errs.append(f"{name}: indices not contiguous from 0")
        for r in rows:
            vals = [float(r[k]) for k in ("plaquette", "P0", "P1", "P2", "P3")]
            if not all(math.isfinite(v) for v in vals):
                errs.append(f"{name}: non-finite value at {idx}={r[idx]}")
                break
            if any(abs(v) > 1 + 1e-12 for v in vals):
                errs.append(f"{name}: |value| > 1 at {idx}={r[idx]}")
                break
    if log:
        txt = open(log).read()
        m = re.search(r"(?:plafond_atteint|caps)=(\d+)", txt)
        t = re.search(r"(?:essais_par_lien_hb|trials_per_link_hb)=([0-9.eE+-]+)", txt)
        if not m:
            errs.append("log: no PERF line with the trial-cap counter")
        elif int(m.group(1)) != 0:
            errs.append(f"log: heat-bath trial cap reached {m.group(1)} times")
        if t and not (1.0 <= float(t.group(1)) <= 2.0):
            errs.append(f"log: trials per link = {t.group(1)} outside [1, 2]")
    return errs


def main():
    args = sys.argv[1:]
    nbad, i = 0, 0
    while i < len(args):
        d, log = args[i], None
        if i + 2 < len(args) and args[i + 1] == "--log":
            log = args[i + 2]
            i += 3
        else:
            i += 1
        e = check(d, log)
        print(f"CHECK_RUN {d}: " + ("OK" if not e else "FAIL " + "; ".join(e)))
        nbad += bool(e)
    print("CHECK_RUN", "PASS" if nbad == 0 else f"FAIL ({nbad} run(s))")
    sys.exit(nbad)


if __name__ == "__main__":
    main()
