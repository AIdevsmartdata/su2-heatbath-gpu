# Use of AI in this repository

## Who wrote what

- The generator (`src/su2_hip.cpp`, September 2026), its mechanical CUDA port (`src/hip2cuda.h`,
  `tools/port.sh`), the CPU programs (`cpu/su2_omp.cpp`, `reference/su2_metropolis_ref.cpp`), the
  tests, the scripts and the documentation were written by AI coding agents: large language models
  working in agentic coding environments, with access to a shell, a compiler and the GPUs.
- The preparation of this repository on 25 October 2026 (English translation of the comments,
  removal of internal notes, CPU programs, test suite, benchmarks, documentation) was done by an
  Anthropic Claude model (Claude Opus 5.5) in Claude Code. The models used for the September 2026
  code were not recorded file by file.
- The author, Kévin Rémondière, directed the work: he set the goals, the requirements (exact
  heat bath, double precision, bit reproducibility, criteria frozen before the runs) and the
  decision to publish, and he is responsible for the release.

## How correctness is established

Nothing in this repository relies on a model having read, reviewed or "checked" the code. The
evidence is mechanical and can be re-run:

1. known answers: Philox known-answer vectors; exact moments of the heat-bath sampler; the
   strong-coupling value of the plaquette at small beta (`tests/`, criteria frozen beforehand in
   `tests/CRITERIA.md`);
2. an independent reference implementation (`reference/su2_metropolis_ref.cpp`) that shares no
   implementation choice with the GPU code (matrices instead of quaternions, Metropolis instead of
   heat bath, Mersenne Twister instead of Philox);
3. bit-level reproducibility on the same machine, and bit-level anchors to archived runs on the two
   reference platforms (`tests/anchor.sh`);
4. statistical comparison with published plaquette values (September 2026, see
   `docs/VALIDATION.md`).

Two lessons shaped this setup. In our experience, agreement between two language models has not
been evidence: errors were caught by an independent computation, a larger test case or a deliberate
mutation, not by a second model reading the work. And a test that passes must be checked to test
the right thing (for example, test T8 compares the initial configurations of the GPU and CPU
programs, to make sure that the CPU baseline really runs the same algorithm).

## Human review

The author reads and approves the README, this file and the documentation before every release.
Throughput figures and validation numbers are copied from the logs of the runs, not estimated.
