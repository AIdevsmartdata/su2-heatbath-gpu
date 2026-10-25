# Benchmarks

`bench/run_bench.sh hip|cuda|cpu [threads...]` (or `make bench-hip`, `make bench-cuda`,
`make bench-cpu`) writes a raw log to `bench/out/` with the machine context (CPU model, number of
CPUs available to the process, load average before and after, available memory, compiler or driver
version) followed by the `BENCH` lines of the programs.

`results/` holds the raw logs behind the table of the main README (25 October 2026):

| file | device | conditions |
|---|---|---|
| `2026-10-25_gpu_radeon8060s_gfx1151_hip.txt` | AMD Radeon 8060S (gfx1151), HIP 7.1 | GPU otherwise idle; host CPU load about 12 from other jobs |
| `2026-10-25_gpu_rtx5060ti_sm120_cuda.txt` | NVIDIA GeForce RTX 5060 Ti (sm_120), driver 595.84 | idle machine |
| `2026-10-25_cpu_ryzen_ai_max_395_omp_loaded.txt` | AMD Ryzen AI Max+ 395, `cpu/su2_omp`, 1/8/16/32 threads | about 12 other threads running at lower priority: pessimistic; the 32-thread lines collapsed under that load and are not representative |
| `2026-10-25_cpu_i5_14600kf_omp.txt` | Intel Core i5-14600KF, `cpu/su2_omp`, 1 and 8 threads on efficiency cores, then 17 threads | idle machine; 3 logical CPUs excluded by local policy |

In these logs the field labelled `online_cpus` is the output of `nproc`, i.e. the number of CPUs
available to the process (17 on the i5 machine because of the affinity mask); the script now
prints both numbers. GPU timings use device events around the sweeps only; CPU timings use a steady
clock around the sweeps. Neither includes measurements, file output or start-up.
