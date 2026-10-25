#!/usr/bin/env bash
# run_tests.sh — runs the tests frozen in tests/CRITERIA.md (L = 6, a few minutes).
# Usage: tests/run_tests.sh hip|cuda|cpu
#   hip | cuda : T0-T10 with bin/su2_hip or bin/su2_cuda (needs `make cpu` too: bin/su2_omp, bin/su2_metropolis_ref)
#   cpu        : CPU-only tests (T0 on su2_omp, T9, T10, C1, C2); no GPU needed
# Environment: PYTHON (default python3, needs numpy), OMP_THREADS (default 4) for the CPU programs.
# Output in tests/out/<backend>/ ; one PASS/FAIL line per test ; exit code = number of failures.
set -u
cd "$(dirname "$0")/.."
B=${1:?usage: tests/run_tests.sh hip|cuda|cpu}
PY=${PYTHON:-python3}
NT=${OMP_THREADS:-4}
O=tests/out/$B
case $B in hip) G=bin/su2_hip;; cuda) G=bin/su2_cuda;; cpu) G=;; *) echo "unknown backend $B"; exit 2;; esac
for x in $G bin/su2_omp bin/su2_metropolis_ref; do [ -x "$x" ] || { echo "missing $x (run make)"; exit 2; }; done
rm -rf "$O"; mkdir -p "$O"
NF=0; NP=0
pass() { echo "PASS $1"; NP=$((NP + 1)); }
fail() { echo "FAIL $1"; NF=$((NF + 1)); }
last() { tail -n 1 "$1"; }
HASH=$(sha256sum src/su2_hip.cpp | cut -d' ' -f1)
OMP="bin/su2_omp --threads $NT"
REF=bin/su2_metropolis_ref
chain() {  # binary outdir beta seed therm nmeas [extra...]
  local bin=$1 d=$2 b=$3 s=$4 th=$5 nm=$6; shift 6
  $bin --L 6 --beta "$b" --seed "$s" --start hot --therm "$th" --nmeas "$nm" --skip 1 --nor 4 \
       --source-hash "$HASH" --outdir "$d" "$@" > "$d.log" 2>&1
}
same_files() {  # dirA dirB [glob]: every file of A (matching glob) exists in B and is byte-identical
  local a=$1 b=$2 g=${3:-*} f n=0
  for f in "$a"/$g; do [ -e "$f" ] || return 1; cmp -s "$f" "$b/$(basename "$f")" || { echo "differs: $(basename "$f")"; return 1; }; n=$((n + 1)); done
  [ "$(ls "$a" | wc -l)" = "$(ls "$b" | wc -l)" ] || [ "$g" != "*" ] || return 1
  echo "$n files identical"
}
echo "# run_tests.sh backend=$B source_sha256=$HASH date=$(date -u +%FT%TZ)"

# T10 — the CUDA port is the mechanical one
if tools/port.sh --check > "$O/t10.log" 2>&1; then pass "T10 $(last "$O/t10.log")"; else fail "T10 $(last "$O/t10.log")"; fi

if [ -n "$G" ]; then
  # T0 — Philox known-answer vectors + u01 bounds
  if $G --selftest > "$O/t0.log" 2>&1 && grep -q "SELFTEST OK" "$O/t0.log"; then pass "T0 selftest ($G)"; else fail "T0 selftest ($G)"; fi
  # T1 — exact heat-bath sampler
  if $PY tests/test_heatbath.py "$G" --out "$O/t1.json" > "$O/t1.log" 2>&1; then pass "T1 $(last "$O/t1.log")"; else fail "T1 $(last "$O/t1.log")"; fi
  # T2 — strong coupling beta = 0.5
  chain "$G" "$O/t2_b0.5" 0.5 2510001 200 20000
  if $PY tests/plaquette_tests.py strong T2 "$O/t2_b0.5" --beta 0.5 > "$O/t2.log" 2>&1; then pass "$(last "$O/t2.log")"; else fail "$(last "$O/t2.log")"; fi
  # T3 — regression against archived L = 6 values
  chain "$G" "$O/t3_b2.3" 2.3 2510023 2000 10000
  chain "$G" "$O/t3_b2.5" 2.5 2510025 2000 10000
  if $PY tests/plaquette_tests.py regress T3_b2.3 "$O/t3_b2.3" --value 0.6029637 --err 0.0000534 > "$O/t3a.log" 2>&1; then pass "$(last "$O/t3a.log")"; else fail "$(last "$O/t3a.log")"; fi
  if $PY tests/plaquette_tests.py regress T3_b2.5 "$O/t3_b2.5" --value 0.6529010 --err 0.0000310 > "$O/t3b.log" 2>&1; then pass "$(last "$O/t3b.log")"; else fail "$(last "$O/t3b.log")"; fi
  # T4 — independent CPU reference (Metropolis, matrices, Mersenne Twister)
  $REF --L 6 --beta 2.3 --seed 7 --hits 8 --therm 500 --nmeas 4000 --out "$O/ref_b2.3.csv" > "$O/ref_b2.3.log" 2>&1
  if $PY tests/plaquette_tests.py compare T4 "$O/t3_b2.3" "$O/ref_b2.3.csv" > "$O/t4.log" 2>&1; then pass "$(last "$O/t4.log")"; else fail "$(last "$O/t4.log")"; fi
  # T5 — bit reproducibility on the same machine
  chain "$G" "$O/t5_a" 2.5 11 300 100 --dump-every 50 --dump-initial
  chain "$G" "$O/t5_b" 2.5 11 300 100 --dump-every 50 --dump-initial
  if r=$(same_files "$O/t5_a" "$O/t5_b"); then pass "T5 bit-reproducible: $r"; else fail "T5 $r"; fi
  # T6 — NPY integrity
  if $PY tests/check_npy.py "$O/t5_a" > "$O/t6.log" 2>&1; then pass "T6 $(last "$O/t6.log")"; else fail "T6 $(last "$O/t6.log")"; fi
  # T7 — run sentinel on every chain
  if $PY tests/check_run.py "$O/t2_b0.5" --log "$O/t2_b0.5.log" "$O/t3_b2.3" --log "$O/t3_b2.3.log" "$O/t3_b2.5" --log "$O/t3_b2.5.log" \
       "$O/t5_a" --log "$O/t5_a.log" "$O/t5_b" --log "$O/t5_b.log" > "$O/t7.log" 2>&1; then pass "T7 $(last "$O/t7.log")"; else fail "T7 $(last "$O/t7.log")"; fi
  # T8 — same algorithm on the CPU: identical initial configuration, then chaotic divergence (info)
  chain "$G" "$O/t8_gpu" 2.5 11 0 5 --dump-every 1 --dump-initial
  chain "$OMP" "$O/t8_cpu" 2.5 11 0 5 --dump-every 1 --dump-initial
  if $PY tests/compare_configs.py "$O/t8_gpu" "$O/t8_cpu" > "$O/t8.log" 2>&1; then pass "T8 $(last "$O/t8.log")"; else fail "T8 $(last "$O/t8.log")"; fi
fi

# T9 — CPU port: output independent of the number of threads
chain "bin/su2_omp --threads 1" "$O/t9_1" 2.5 11 300 100 --dump-every 50 --dump-initial
chain "bin/su2_omp --threads $NT" "$O/t9_n" 2.5 11 300 100 --dump-every 50 --dump-initial
if r=$(same_files "$O/t9_1" "$O/t9_n" "*.npy") && r2=$(same_files "$O/t9_1" "$O/t9_n" "*.csv"); then pass "T9 threads 1 vs $NT: npy $r, csv $r2"; else fail "T9 $r ${r2:-}"; fi

if [ -z "$G" ]; then
  if bin/su2_omp --selftest > "$O/t0.log" 2>&1 && grep -q "SELFTEST OK" "$O/t0.log"; then pass "T0 selftest (bin/su2_omp)"; else fail "T0 selftest (bin/su2_omp)"; fi
  # C1 — strong coupling with the two CPU programs
  chain "$OMP" "$O/c1_omp" 0.5 2510001 200 5000
  if $PY tests/plaquette_tests.py strong C1_omp "$O/c1_omp" --beta 0.5 > "$O/c1a.log" 2>&1; then pass "$(last "$O/c1a.log")"; else fail "$(last "$O/c1a.log")"; fi
  $REF --L 6 --beta 0.5 --seed 5 --hits 8 --therm 500 --nmeas 4000 --out "$O/ref_b0.5.csv" > "$O/ref_b0.5.log" 2>&1
  if $PY tests/plaquette_tests.py strong C1_ref "$O/ref_b0.5.csv" --beta 0.5 > "$O/c1b.log" 2>&1; then pass "$(last "$O/c1b.log")"; else fail "$(last "$O/c1b.log")"; fi
  # C2 — su2_omp against the independent reference at beta = 2.3
  chain "$OMP" "$O/c2_omp" 2.3 2510023 2000 5000
  $REF --L 6 --beta 2.3 --seed 7 --hits 8 --therm 500 --nmeas 4000 --out "$O/ref_b2.3.csv" > "$O/ref_b2.3.log" 2>&1
  if $PY tests/plaquette_tests.py compare C2 "$O/c2_omp" "$O/ref_b2.3.csv" > "$O/c2.log" 2>&1; then pass "$(last "$O/c2.log")"; else fail "$(last "$O/c2.log")"; fi
fi

echo "TESTS backend=$B pass=$NP fail=$NF"
exit $NF
