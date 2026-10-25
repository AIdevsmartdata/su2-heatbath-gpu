#!/usr/bin/env bash
# run_bench.sh — throughput in link updates per second (FP64), with the machine context.
# Usage: bench/run_bench.sh hip|cuda            GPU: --bench at L = 8, 16, 24, 32 (beta = 2.5), 3 repetitions
#        bench/run_bench.sh cpu [T1 T2 ...]     CPU (cpu/su2_omp): L = 16, 24, 32 with the given thread counts
#                                               (default: 1 and the number of online CPUs)
# The GPU timing uses device events around the sweeps only (no host I/O); the CPU timing uses a
# steady clock around the sweeps. A "link update" is one heat-bath (HB) or overrelaxation (OR) update
# of one link; one Monte Carlo step of the production protocol is 1 HB + 4 OR sweeps.
# Results: bench/out/<backend>_<UTC timestamp>.txt (raw lines + context). Report the machine load:
# a loaded machine gives lower numbers.
set -u
cd "$(dirname "$0")/.."
B=${1:?usage: bench/run_bench.sh hip|cuda|cpu [threads...]}; shift || true
mkdir -p bench/out
OUT=bench/out/${B}_$(date -u +%Y%m%dT%H%M%SZ).txt
{
  echo "# date_utc $(date -u +%FT%TZ)"
  echo "# cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//') ; cpus_available_to_this_process $(nproc) of $(getconf _NPROCESSORS_ONLN)"
  echo "# loadavg_before $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "# mem_available_GiB $(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)"
  case $B in
    hip)  echo "# hipcc $(hipcc --version 2>/dev/null | head -1)";;
    cuda) echo "# nvidia-smi $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)";;
    cpu)  echo "# cxx $(${CXX:-g++} --version | head -1)";;
  esac
} > "$OUT"
case $B in
  hip|cuda)
    G=bin/su2_$B
    for L in 8 16 24 32; do
      ns=200; [ "$L" -ge 24 ] && ns=100
      for rep in 1 2 3; do $G --bench --L "$L" --beta 2.5 --nsweeps $ns >> "$OUT" 2>&1; done
    done;;
  cpu)
    TH=${*:-"1 $(nproc)"}
    for T in $TH; do
      for L in 16 24 32; do
        ns=100; [ "$L" = 24 ] && ns=40; [ "$L" = 32 ] && ns=16; [ "$T" = 1 ] && ns=$((ns / 4 > 4 ? ns / 4 : 4))
        bin/su2_omp --bench --L "$L" --beta 2.5 --nsweeps $ns --threads "$T" >> "$OUT" 2>&1
      done
    done;;
  *) echo "unknown backend $B"; exit 2;;
esac
echo "# loadavg_after $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT"
cat "$OUT"
echo "written $OUT"
