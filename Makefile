# su2-heatbath-gpu — build and test.
#   make hip | cuda | cpu        build bin/su2_hip, bin/su2_cuda, or the CPU programs (bin/su2_omp, bin/su2_metropolis_ref)
#   make test-hip | test-cuda    GPU tests of tests/CRITERIA.md (L = 6, a few minutes; needs python3 + numpy)
#   make test-cpu                CPU-only tests (no GPU)
#   make anchor-hip | anchor-cuda   bit-level comparison with archived runs (reference platforms only)
#   make bench-hip | bench-cuda | bench-cpu   throughput measurements (bench/out/)
#   make port | port-check       regenerate / check src/su2_cuda.cu and src/port.diff
# Variables: HIPCC, HIP_ARCH (default gfx1151), NVCC, CUDA_ARCH (default sm_120), CXX, PYTHON.

HIPCC     ?= hipcc
HIP_ARCH  ?= gfx1151
NVCC      ?= nvcc
CUDA_ARCH ?= sm_120
CXX       ?= g++
PYTHON    ?= python3
export PYTHON

.PHONY: help all hip cuda cpu port port-check test-hip test-cuda test-cpu anchor-hip anchor-cuda bench-hip bench-cuda bench-cpu clean

help:
	@sed -n '2,9p' Makefile | sed 's/^# \{0,1\}//'

# The HIP command line is the one used for the archived production binary: same flags, source given
# as src/su2_hip.cpp from the repository root (the file name enters the error messages).
hip: bin/su2_hip
bin/su2_hip: src/su2_hip.cpp
	@mkdir -p bin
	$(HIPCC) --offload-arch=$(HIP_ARCH) -O3 -std=c++17 -o bin/su2_hip src/su2_hip.cpp

cuda: bin/su2_cuda
bin/su2_cuda: src/su2_cuda.cu src/hip2cuda.h
	@mkdir -p bin
	cd src && $(NVCC) -arch=$(CUDA_ARCH) -O3 -std=c++17 -o ../bin/su2_cuda su2_cuda.cu

cpu: bin/su2_omp bin/su2_metropolis_ref
bin/su2_omp: cpu/su2_omp.cpp
	@mkdir -p bin
	$(CXX) -O3 -march=native -std=c++17 -fopenmp -o bin/su2_omp cpu/su2_omp.cpp
bin/su2_metropolis_ref: reference/su2_metropolis_ref.cpp
	@mkdir -p bin
	$(CXX) -O2 -std=c++17 -o bin/su2_metropolis_ref reference/su2_metropolis_ref.cpp

port:
	tools/port.sh
port-check:
	tools/port.sh --check

test-hip: hip cpu
	tests/run_tests.sh hip
test-cuda: cuda cpu
	tests/run_tests.sh cuda
test-cpu: cpu
	tests/run_tests.sh cpu

anchor-hip: hip
	tests/anchor.sh hip
anchor-cuda: cuda
	tests/anchor.sh cuda

bench-hip: hip
	bench/run_bench.sh hip
bench-cuda: cuda
	bench/run_bench.sh cuda
bench-cpu: bin/su2_omp
	bench/run_bench.sh cpu

clean:
	rm -rf bin tests/out bench/out
