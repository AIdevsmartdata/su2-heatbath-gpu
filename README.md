# su2-heatbath-gpu

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23644086.svg)](https://doi.org/10.5281/zenodo.23644086)

Archived on Zenodo: all versions [doi:10.5281/zenodo.23644086](https://doi.org/10.5281/zenodo.23644086); v1.0.0 [doi:10.5281/zenodo.23644087](https://doi.org/10.5281/zenodo.23644087).


A small, self-contained Monte Carlo generator of SU(2) pure-gauge configurations with the Wilson
action on a periodic four-dimensional lattice, in double precision, for GPUs. One HIP source file
(tested on an AMD Radeon 8060S integrated GPU, gfx1151, ROCm 7.1) and its mechanical CUDA port
(tested on an NVIDIA GeForce RTX 5060 Ti, sm_120, CUDA 13.3). Each Monte Carlo step is one exact
heat-bath sweep (Creutz / Kennedy-Pendleton) plus overrelaxation sweeps, with checkerboard
updates and counter-based Philox4x32-10 random numbers: on a given GPU and build, a run is
reproducible bit for bit.

## What it is

- A generator of quenched SU(2) ensembles: configurations as NumPy `.npy` files, plaquette and
  Polyakov-loop histories as CSV, a JSON manifest with all parameters.
- Small: `src/su2_hip.cpp` (638 lines), plus a header of 20 macros (`src/hip2cuda.h`) for the
  CUDA build. The CUDA file `src/su2_cuda.cu` is generated from the HIP file by `tools/port.sh`; the
  whole difference is 4 changed lines and 2 comment lines (`src/port.diff`).
- Tested: a suite of known-answer and cross-check tests at L = 6 that runs in about a minute
  (`tests/`, criteria frozen in `tests/CRITERIA.md`), an independent CPU reference code, and two
  bit-level anchors to archived runs (see [Validation](#validation)).

## What it is not

- Not a lattice QCD package: no fermions, no other gauge groups, no smearing or gradient flow, no
  Wilson loops, static potential or glueball measurements, no anisotropic lattices, a single GPU
  only, no checkpoint and restart, no single-precision mode.
- Not new physics and not a new algorithm. SU(2) heat-bath simulations on GPUs have been published
  since at least 2010 (N. Cardoso and P. Bicudo, arXiv:1010.4834), and larger open-source codes
  cover far more ground. What this repository offers is a short, readable, reproducible tool with
  its tests.
- Not tuned for speed: one GPU thread per link, everything in FP64, no auto-tuning.

## Repository layout

| path | content |
|---|---|
| `src/` | `su2_hip.cpp` (the generator), `su2_cuda.cu` + `hip2cuda.h` (generated CUDA port), `port.diff` |
| `tools/port.sh` | regenerates (or checks) the CUDA port |
| `cpu/su2_omp.cpp` | the same algorithm on the CPU with OpenMP (baseline, CPU-only tests) |
| `reference/su2_metropolis_ref.cpp` | independent CPU reference (Metropolis) |
| `tests/` | test driver, frozen criteria, run and dump checkers, platform anchors |
| `bench/` | benchmark script and raw results |
| `docs/` | algorithm, conventions and formats; validation |

## Quick start

```sh
make hip            # bin/su2_hip   (set HIP_ARCH for a GPU other than gfx1151)
make cuda           # bin/su2_cuda  (set CUDA_ARCH, e.g. CUDA_ARCH=sm_86, and NVCC if needed)
make cpu            # bin/su2_omp and bin/su2_metropolis_ref (used by the tests and the CPU baseline)
make test-hip       # or: make test-cuda ; about one minute; needs python3 + numpy
make test-cpu       # CPU-only subset, no GPU needed

./bin/su2_hip --L 16 --beta 2.5 --seed 123 --start hot --therm 3000 --nmeas 2000 --skip 1 \
    --nor 4 --dump-every 100 --outdir run_L16 --source-hash "$(sha256sum src/su2_hip.cpp | cut -d' ' -f1)"
python3 tests/check_run.py run_L16      # always check a run before using it (see below)
```

Requirements: ROCm with `hipcc` (tested: HIP 7.1.52801, clang 21.1.8) or the CUDA toolkit (sm_120
needs CUDA 12.8 or later; tested: 13.3, driver 595.84); a GPU with FP64 support; GCC with OpenMP
for the CPU programs (tested: GCC 15.2); Python 3 with NumPy for the tests (tested: Python 3.14,
NumPy 2.5).
Other GPU architectures should work but have not been tested. Memory: 32 bytes per link, i.e.
128·L⁴ bytes for the gauge field (L = 32: 134 MB; L = 64: 2.1 GB), plus small buffers.

## Usage

| option | meaning |
|---|---|
| `--L N` | lattice extent, even, at least 2 (L⁴ sites, 4·L⁴ links) |
| `--beta B` | coupling of S = beta Σ_p (1 − ½ Re Tr U_p); must be > 0 |
| `--seed S` | 64-bit integer, the Philox key |
| `--start hot\|cold` | Haar-random or unit initial field |
| `--therm N` | thermalisation steps (every step is measured and written to `therm.csv`) |
| `--nmeas N`, `--skip K` | N measurements, K steps apart |
| `--nor N` | overrelaxation sweeps per step (one step = 1 heat-bath sweep + N OR sweeps) |
| `--dump-every M`, `--dump-initial` | write the configuration every M measurements, and the initial one |
| `--outdir DIR` | output directory (created if needed) |
| `--source-hash H` | free text copied into the manifest; we pass the sha256 of `src/su2_hip.cpp` |
| `--selftest` | Philox known-answer vectors and bounds of the uniform generator (no GPU needed) |
| `--bench --nsweeps N` | timing only: N heat-bath then N OR sweeps, reports link updates per second |
| `--test-hb ALPHA N` | N draws of the heat-bath variable x0 at fixed ALPHA (used by tests/test_heatbath.py) |

Outputs in `--outdir`:

- `therm.csv`: `step,plaquette,P0,P1,P2,P3` (step 0 is the initial state);
- `measurements.csv`: `measurement_index,total_steps,phase,plaquette,P0,P1,P2,P3` (index 0 is the
  end of thermalisation, `phase` is `initial` or `sample`);
- `cfg_XXXXXX.npy`: complex128, C order, shape (4, L, L, L, L, 2, 2) = (mu, x0, x1, x2, x3, i, j),
  direction 3 is the temporal one; each 2×2 block is a unitary matrix of determinant 1;
- `manifest.json`: parameters; its `"statut"` field goes from `EN_COURS` to `COMPLETE`, and an
  empty `COMPLETE` file is written at the very end.

The plaquette is the average of ½ Re Tr U_p over the 6·L⁴ plaquettes; P_mu is the average over
transverse sites of ½ Tr of the Polyakov line in direction mu. Exit codes: 0 success, 2 bad
arguments, 3 GPU runtime error, 4 or 5 file error, 6 heat-bath trial cap reached (no `COMPLETE`),
7 internal history buffer full. Runtime messages and a few output keys are in French; a glossary is
in [docs/ALGORITHM.md](docs/ALGORITHM.md), with the conventions, the random-number counters and
the file formats.

## Measured throughput

Link updates per second, FP64, beta = 2.5, `--bench` (timing of the sweeps only, device events on
the GPUs), median of three runs on the GPUs and single runs on the CPUs, 25 October 2026. "HB" = heat bath, "OR" =
overrelaxation; a production step (1 HB + 4 OR sweeps, with a measurement and CSV output every
step) is slower than the sum of the sweeps at small L. Raw logs: [bench/results/](bench/results/).

| device, precision, build | L = 16 HB | L = 16 OR | L = 32 HB | L = 32 OR |
|---|---|---|---|---|
| AMD Radeon 8060S (integrated, gfx1151), FP64, HIP 7.1 | 1.57e8 | 3.68e8 | 1.81e8 | 4.13e8 |
| NVIDIA GeForce RTX 5060 Ti (sm_120), FP64, CUDA 13.3 | 2.18e8 | 4.57e8 | 2.63e8 | 5.33e8 |
| same algorithm on the CPU (`cpu/su2_omp`), AMD Ryzen AI Max+ 395, 16 threads, machine loaded (see below) | 7.20e7 | 1.40e8 | 7.41e7 | 1.16e8 |
| same algorithm on the CPU, Intel Core i5-14600KF, 17 threads, idle machine | 8.64e7 | 1.71e8 | 6.71e7 | 6.85e7 |
| single CPU thread, Ryzen AI Max+ 395 / i5-14600KF (efficiency core) | 8.3e6 / 5.7e6 | 1.9e7 / 1.2e7 | 7.9e6 / 5.7e6 | 1.5e7 / 1.1e7 |

How to read this table:

- The CPU baseline is a plain OpenMP transcription of the same algorithm with the same random
  streams (test T8 checks that both start from the same configuration), compiled with
  `g++ -O3 -march=native`: no hand vectorisation, no layout tuning. A tuned CPU code would likely
  be faster.
- The Ryzen numbers were taken while about twelve other threads were running at lower priority
  (load average about 12 on 32 logical CPUs before the measurement); they are therefore
  pessimistic, and 32 threads were much slower than 16 under that load (OpenMP barriers), so they
  are not shown. The GPU numbers of that machine were taken under the same CPU load (the GPU itself
  was otherwise idle). On the i5 machine (idle), 3 of the 20 logical CPUs were excluded by local
  policy.
- On these two machines, for this kernel in FP64, the GPU is about 2 to 3.5 times faster than the
  CPU of the same machine running the same algorithm at L = 16-24. At L = 32 the i5 CPU becomes
  limited by memory bandwidth and the ratio grows (3.9 for HB, 7.8 for OR). We do not quote
  speed-ups against single-threaded code.
- Small lattices are latency-bound on the GPUs. In the L = 6 test runs (production steps, with a
  measurement every step), the Radeon 8060S made 1390 to 1570 steps per second, the RTX 5060 Ti
  900 to 1080, and `cpu/su2_omp` with 4 threads on the (loaded) Ryzen 2110 to 2450: for very small
  lattices the CPU program can be the faster choice.

## Validation

Details and all numbers: [docs/VALIDATION.md](docs/VALIDATION.md).

The test suite (`make test-hip` / `make test-cuda`) checks, at L = 6: the Philox generator against
its published known-answer vectors (T0); the exact heat-bath sampler against exact moments, both
branches (T1); the plaquette at beta = 0.5 against the leading strong-coupling result u + 4u⁵,
u = I₂(β)/I₁(β) (T2); the plaquette at beta = 2.3 and 2.5 against archived high-statistics values
(T3); agreement with an independent CPU code that shares no implementation choice: 2×2 complex
matrices, Metropolis updates, Mersenne Twister (T4); bit reproducibility (T5); unitarity and
layout of the dumps (T6); a run sentinel (T7); the CPU transcription starts from the same
configuration (T8) and does not depend on the number of threads (T9); the CUDA file is the
mechanical port (T10).

Results on 25 October 2026: 12/12 tests passed on both GPUs. In addition:

- HIP anchor: built with HIP 7.1 / clang 21.1.8 for gfx1151, the source of this repository
  compiles to a binary byte-identical (sha256 `d62ff7d2…`) to the one that produced a set of
  archived production ensembles in September 2026, and it regenerates one of those chains
  (L = 12, beta = 2.5, 7000 steps) byte for byte;
- CUDA anchor: built with nvcc 13.3 for sm_120, it regenerates byte for byte a chain of the CUDA
  validation of September 2026 (L = 8, beta = 2.8, 22000 steps). nvcc builds themselves are not
  bit-reproducible (two compilations of the same file differ), so only outputs are compared.

Reproducibility has a scope: bit equality holds on the same GPU architecture with the same
compiler and runtime. Across GPUs or compilers the Philox integers are identical, but last-bit
differences in the math library are amplified by the Markov chain, which decorrelates within
about ten steps (test T8 prints the growth); there, compare statistically.

## Silent data corruption on the gfx1151 integrated GPU

Please read this before using the HIP build on an APU. On 4 October 2026, on the AMD Radeon 8060S
integrated GPU of our development machine (it shares the system memory), FP64 runs of this
generator (built from the same source) were corrupted in a transient, non-deterministic way, with
no error reported by the GPU runtime:

- in the morning, 2 of 6 runs at L = 16 were affected: one (beta = 2.5) produced NaN from step 770
  together with 18 heat-bath trial-cap events, which the program catches (exit code 6, no
  `COMPLETE`); the other (beta = 16) diverged from step 1102 and had an isolated NaN in its
  measurement file;
- in the afternoon, a run at L = 16, beta = 16 ended with exit code 0 and a `COMPLETE` marker
  although its plaquette history contained a NaN at measurement 2083;
- every affected chain, re-run with the same arguments, was clean, and bit-identical between
  re-runs. The same day, a statistical sampler test of a related program on this GPU failed
  (z = −6.4) and passed when re-run with the same seeds.

That morning, three GPU faults happened within one hour while several heavy CPU jobs were running,
two of them when the available memory had dropped to 14.7 GB and 7.0 GB; but the afternoon
corruption happened with about 60 to 70 GB available. The cause (hardware, driver or runtime) has
not been diagnosed. We have not observed such corruption on the NVIDIA GPU, which was used less.

How to protect a production run:

1. run `tests/check_run.py` on every output directory (finite values, |plaquette| ≤ 1, `COMPLETE`,
   heat-bath trial counter in the log);
2. check the dumps with `tests/check_npy.py` (unitarity, plaquette recomputed with NumPy);
3. since runs are bit-reproducible on the same GPU and build, re-run (part of) every chain you rely
   on with the same arguments and compare the files with `cmp`: a corrupted run does not reproduce;
4. on an APU, keep plenty of memory free during GPU production (we keep at least 30 GB free, of
   about 123 GB of system memory) and avoid running other GPU workloads at the same time.

The generator has no in-loop sentinel: a fault that does not reach a measured quantity can only be
found by re-running.

## Known limitations

- `--beta 0` is not supported: the Creutz sampler divides by alpha = beta·k, every heat-bath update
  then exhausts its cap of 10⁶ trials (very slow), and the program exits with code 6.
- The RNG counter uses a 32-bit link index, so 4·L⁴ must stay below 2³² (L ≤ 180); the program does
  not check this, but such a lattice would need about 134 GB for the field alone.
- The `backend` field of `manifest.json` is a fixed string (`HIP gfx1151 FP64`, or `CUDA sm_120 FP64
  (portage de su2_hip)`), whatever architecture the binary was built for; the actual device is
  printed on standard error at start-up. `source_hash` is whatever `--source-hash` received
  (`non_fourni`, "not provided", by default).
- No checkpoint or restart: a run must complete in one go.
- The host must be little-endian (NPY header), as x86-64 is.
- nvcc prints one warning (an unused host function `polyakov`); it is harmless.

## How this code was written

The code, the tests and this documentation were written by AI coding agents working under the
direction of the author, who chose the goals and the validation protocol; correctness rests on
mechanical checks (tests with frozen criteria, an independent reference code, bit-level anchors),
not on reading by a model. See [AI_USE.md](AI_USE.md).

## References

- M. Creutz, Monte Carlo study of quantized SU(2) gauge theory, Phys. Rev. D 21 (1980) 2308,
  doi:10.1103/PhysRevD.21.2308 (heat bath for SU(2)).
- A. D. Kennedy and B. J. Pendleton, Improved heatbath method for Monte Carlo calculations in
  lattice gauge theories, Phys. Lett. B 156 (1985) 393, doi:10.1016/0370-2693(85)91632-6.
- M. Creutz, Overrelaxation and Monte Carlo simulation, Phys. Rev. D 36 (1987) 515,
  doi:10.1103/PhysRevD.36.515; F. R. Brown and T. J. Woch, Phys. Rev. Lett. 58 (1987) 2394,
  doi:10.1103/PhysRevLett.58.2394 (overrelaxation).
- J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, Parallel random numbers: as easy as
  1, 2, 3, SC '11 (2011), doi:10.1145/2063384.2063405 (Philox).
- A. Athenodorou and M. Teper, SU(N) gauge theories in 3+1 dimensions: glueball spectrum, string
  tensions and topology, JHEP 12 (2021) 082, doi:10.1007/JHEP12(2021)082, arXiv:2106.00364
  (plaquette values used in the September 2026 validation).
- N. Cardoso and P. Bicudo, SU(2) lattice gauge theory simulations on Fermi GPUs,
  arXiv:1010.4834 (earlier GPU work).

## Citing and license

Please cite the archived release (see [CITATION.cff](CITATION.cff); the DOI will be added at the
first release). Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE)); see also
[NOTICE](NOTICE).
