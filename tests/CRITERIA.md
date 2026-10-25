# Test criteria (frozen before the first run)

These criteria were written on 2026-10-25, before any of the tests below was run on the code of
this repository, and they are not adjusted after looking at the results. Seeds are fixed, so on a
given machine and build every statistical test gives the same answer each time it is run.

All lattices are L = 6 (small and fast). One Monte Carlo step = 1 heat-bath sweep + 4
overrelaxation sweeps (`--nor 4`). Statistical errors use the integrated autocorrelation time with
Sokal's automatic window (c = 6): error = sqrt(2 tau_int var / n).

| id | what | pass criterion |
|---|---|---|
| T0 | `--selftest`: Philox4x32-10 against the three Random123 known-answer vectors, bounds of the uniform generator | prints `SELFTEST OK`, rc 0 |
| T1 | exact heat-bath sampler: `--test-hb ALPHA 2e7` for ALPHA in {0.3, 1, 1.999 (Creutz); 2, 5, 20, 100 (Kennedy-Pendleton)}, seeds 9000+i | first two moments of x0 within \|z\| < 4 of the exact moments (Gauss-Chebyshev quadrature, 4000 nodes); trial cap never reached |
| T2 | strong coupling, beta = 0.5: hot start, seed 2510001, 200 + 20000 steps | \|P - (u + 4u^5)\| < 4 sigma_tot, u = I2(beta)/I1(beta), sigma_tot^2 = sigma_stat^2 + (50 u^7)^2 (see docs/VALIDATION.md for the expansion) |
| T3 | regression against archived high-statistics values (L = 6, 3 chains each, produced with the same source in September 2026 and cross-checked then against an independent CPU code): beta = 2.3 -> 0.6029637(53), beta = 2.5 -> 0.6529010(31); one hot chain each, seeds 2510023 / 2510025, 2000 + 10000 steps | \|z\| < 3 with the combined error |
| T4 | independent CPU reference (`reference/su2_metropolis_ref`: matrices, Metropolis, Mersenne Twister) at beta = 2.3, seed 7, 8 hits, 500 + 4000 sweeps, against the T3 beta = 2.3 chain | \|z\| < 3 with the combined error |
| T5 | bit reproducibility: the same run (beta = 2.5, seed 11, hot, 300 + 100 steps, dumps every 50 steps and the initial one) executed twice | every output file byte-identical (`cmp`) |
| T6 | NPY integrity of the T5 dumps | shape (4, L, L, L, L, 2, 2), complex128, C order; max\|UU^+ - 1\| < 1e-13; max\|det U - 1\| < 1e-13; SU(2) structure < 1e-13; plaquette recomputed by numpy = CSV value at 1e-12 |
| T7 | run sentinel on every chain of T2, T3, T5 | `COMPLETE` present, manifest status COMPLETE, every CSV value finite, \|plaquette\| <= 1, \|P_mu\| <= 1, contiguous indices, heat-bath trial cap never reached |
| T8 | same algorithm on the CPU (`cpu/su2_omp`): GPU and CPU runs with identical arguments (beta = 2.5, seed 11, hot) | initial configurations agree to max\|dU\| < 1e-13 (same Philox counters; only libm rounding differs). The divergence after 1..5 steps is printed for information |
| T9 | CPU port independent of the thread count: `su2_omp` with 1 and 4 threads | every CSV and NPY file byte-identical |
| T10 | `tools/port.sh --check` | `src/su2_cuda.cu` and `src/port.diff` match the generated port |
| C1 | CPU-only strong coupling: `su2_omp` (beta = 0.5, seed 2510001, 200 + 5000 steps) and the Metropolis reference (beta = 0.5, seed 5, 500 + 4000 sweeps) | same criterion as T2 for each |
| C2 | CPU-only cross-check: `su2_omp` (beta = 2.3, seed 2510023, 2000 + 5000 steps) against the Metropolis reference of T4 | \|z\| < 3 |

Platform anchors (informational outside the reference platforms, see docs/VALIDATION.md):

| id | what | expected |
|---|---|---|
| A-HIP | `bin/su2_hip` built with HIP 7.1 / clang 21.1.8 for gfx1151; archived production chain L = 12, beta = 2.5, seed 26095612011, 5000 + 2000 steps | sha256 of the binary and of `cfg_002000.npy`, `measurements.csv`, `therm.csv` equal to the archived ones |
| A-CUDA | `bin/su2_cuda` built with nvcc 13.3 for sm_120; archived chain L = 8, beta = 2.8 of the CUDA port validation | sha256 of the output files equal to the archived ones |

Thresholds: \|z\| < 3 for comparisons between two Monte Carlo estimates, \|z\| < 4 where an exact
or analytic value is involved (as in the original September 2026 sampler test).
