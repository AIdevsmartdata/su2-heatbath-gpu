#!/usr/bin/env bash
# anchor.sh — platform anchors (informational outside the reference platforms).
# Re-runs one archived chain and compares sha256 sums with tests/anchors.sha256:
#   hip  : gfx1151, HIP 7.1 / clang 21.1.8 — production chain L=12, beta=2.5 (September 2026), and the binary itself
#   cuda : sm_120, nvcc 13.3 — chain L=8, beta=2.8 of the CUDA port validation (September 2026)
# Bit equality is only expected with the same GPU architecture and the same compiler/runtime versions
# (libm rounding differs elsewhere and the Markov chain amplifies it). A mismatch on another platform is
# NOT an error: use tests/run_tests.sh for the statistical checks.
# Usage: tests/anchor.sh hip|cuda      exit code 0 = all anchors match
set -u
cd "$(dirname "$0")/.."
B=${1:?usage: tests/anchor.sh hip|cuda}
O=tests/out/anchor_$B; rm -rf "$O"; mkdir -p "$O"
case $B in
  hip)  ./bin/su2_hip --L 12 --beta 2.5 --seed 26095612011 --start hot --therm 5000 --nmeas 2000 --skip 1 --nor 4 \
          --dump-every 100 --outdir "$O/run" > "$O/run.log" 2>&1; rc=$?; files="cfg_002000.npy measurements.csv therm.csv";;
  cuda) ./bin/su2_cuda --L 8 --beta 2.8 --seed 27092708011 --start hot --therm 2000 --nmeas 20000 --skip 1 --nor 4 \
          --dump-every 1000 --outdir "$O/run" > "$O/run.log" 2>&1; rc=$?; files="cfg_020000.npy measurements.csv therm.csv";;
  *) echo "unknown backend $B"; exit 2;;
esac
[ $rc = 0 ] || { echo "ANCHOR run failed rc=$rc"; exit 1; }
bad=0
exp_bin=$(awk -v b="$B" '$1==b && $2=="binary" {print $3}' tests/anchors.sha256)
got_bin=$(sha256sum "bin/su2_$B" | cut -d' ' -f1)
if [ -n "$exp_bin" ]; then
  [ "$got_bin" = "$exp_bin" ] && echo "ANCHOR binary bin/su2_$B MATCH $got_bin" || echo "ANCHOR binary bin/su2_$B different (got $got_bin, archived $exp_bin) — informational"
fi
for f in $files; do
  e=$(awk -v b="$B" -v f="$f" '$1==b && $2==f {print $3}' tests/anchors.sha256)
  g=$(sha256sum "$O/run/$f" | cut -d' ' -f1)
  if [ "$g" = "$e" ]; then echo "ANCHOR $f MATCH $g"; else echo "ANCHOR $f DIFFERENT got $g archived $e"; bad=$((bad + 1)); fi
done
rm -f "$O"/run/cfg_*.npy.part
echo "ANCHOR $B: $([ $bad = 0 ] && echo 'ALL MATCH (same platform as the archived run)' || echo "$bad mismatch(es)")"
exit $bad
