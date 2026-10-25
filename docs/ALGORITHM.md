# Algorithm, conventions and formats

Everything below describes `src/su2_hip.cpp` (and therefore `src/su2_cuda.cu`, which differs only
in the runtime include and in a few identification strings, see `src/port.diff`).

## Action and representation

- Wilson action S = beta Σ_p (1 − ½ Re Tr U_p), sum over the 6·L⁴ plaquettes of a periodic L⁴
  lattice (isotropic; direction 3 is called temporal).
- Sites: s = ((x0·L + x1)·L + x2)·L + x3. Link U_mu(x) goes from x to x + e_mu, where e_mu
  increments coordinate x_mu. Storage on the GPU: `U[mu·V + s]`, V = L⁴.
- SU(2) elements are unit quaternions (a0, a1, a2, a3), U = a0 + i (a1 σ1 + a2 σ2 + a3 σ3), i.e.
  the matrix [[a0 + i a3, a2 + i a1], [−a2 + i a1, a0 − i a3]]. Products use
  (a0 + i a·σ)(b0 + i b·σ) = (a0 b0 − a·b) + i (a0 b + b0 a − a × b)·σ. Every updated link is
  re-normalised to unit length.

## One Monte Carlo step

One heat-bath sweep followed by `--nor` overrelaxation sweeps. A sweep visits the 4 directions and,
for each, the two checkerboard parities of x; links U_mu(x) with the same direction and the same
parity of x do not appear in each other's staples, so each (mu, parity) batch is updated in
parallel, one GPU thread per link, without any race.

Heat bath for one link: A = sum of the 6 staples, such that the local action is
−(beta/2) Re Tr(U_mu(x) A); k = |A|, V = A / k. Then X = U V is distributed as
exp(alpha x0) dHaar(X) with alpha = beta·k, so x0 = ½ Tr X has the density
sqrt(1 − x0²) exp(alpha x0) on [−1, 1]. x0 is drawn exactly:

- alpha ≥ 2, Kennedy-Pendleton: lambda² = −(ln a1 + cos²(2π a2) ln a3) / (2 alpha), accepted if
  a4² ≤ 1 − lambda², then x0 = 1 − 2 lambda²;
- alpha < 2, Creutz: y = 1 + ln(e^(−2 alpha) + a1 (1 − e^(−2 alpha))) / alpha, accepted if
  a2² ≤ 1 − y², then x0 = y.

At most 10⁶ trials per link: if the cap is reached the link is left unchanged and counted; the
program then exits with code 6 and writes no `COMPLETE` marker. The other three components of X are
uniform on the sphere of radius sqrt(1 − x0²) (cos theta = 2u − 1, phi = 2πu'), and the new link is
U = X V⁺. A zero (or non-finite) staple gives a Haar-random link instead; this has not been
observed in practice.

Overrelaxation: U → V⁺ U⁺ V⁺, which keeps Re Tr(U A) unchanged and uses no random numbers.

Hot start: every link is Haar-random (four Gaussian numbers from two Box-Muller pairs, normalised).
Cold start: every link is the identity.

## Random numbers

Philox4x32-10 (Salmon et al., SC '11), counter-based: every random block is a pure function of a
key and a counter, with no state shared between threads. The key is the 64-bit `--seed` (low and
high 32 bits). The 128-bit counter is

| use | counter (4 × 32 bits) |
|---|---|
| hot start, link i | (i, 0xFFFFFFFF, j, "INIT") with j = 0, 1 |
| heat bath, link li = mu·V + s | (li, sweep, draw, "HBAT"), sweep = index of the heat-bath sweep (from 0), draw = trial counter within the update |
| `--test-hb`, sample i | (i, 0x7E57, draw, "HBAT") |

("INIT" = 0x494E4954, "HBAT" = 0x48424154.) Two 32-bit outputs give one double in ]0, 1[:
((a >> 6)·2²⁶ + (b >> 6) + 0.5)·2⁻⁵², which is never 0 nor 1. The implementation is checked against
the three known-answer vectors distributed with the Random123 library (`--selftest`). Because of
the 32-bit link index, 4·L⁴ must stay below 2³² (not checked by the program).

Consequence: a chain is a deterministic function of (arguments, binary). Reductions (plaquette,
Polyakov loops) use a fixed tree per block of 256 threads followed by a fixed-order final sum, so
the measurements are deterministic too. On another GPU or with another compiler, the Philox
integers are identical but the math library (log, cos, sqrt, exp) may round differently in the
last bit; the Markov chain amplifies such differences and two such chains decorrelate within about
ten steps.

## Measurements and files

- Plaquette: (1/(6V)) Σ_x Σ_{mu<nu} ½ Re Tr U_mu(x) U_nu(x+mu) U_mu(x+nu)⁺ U_nu(x)⁺.
- Polyakov loops: P_mu = average over the L³ transverse sites of ½ Tr Π_{k=0}^{L−1} U_mu(x + k e_mu).
- `therm.csv`: one line per thermalisation step, step 0 being the initial configuration.
- `measurements.csv`: index 0 is the configuration at the end of thermalisation (`phase` =
  `initial`), indices 1..N are taken every `--skip` steps (`phase` = `sample`); `total_steps` counts
  the steps since the start.
- `cfg_XXXXXX.npy`: written at measurement indices that are multiples of `--dump-every` (and at index
  0 = initial configuration with `--dump-initial`); NPY version 1.0, `<c16`, C order, shape
  (4, L, L, L, L, 2, 2); written to `*.part` and renamed when complete.
- `manifest.json`: written at start with status `EN_COURS` and rewritten with `COMPLETE` at the end.

The CPU programs (`cpu/su2_omp`, same algorithm; `reference/su2_metropolis_ref`, independent
Metropolis code) are described in their headers. `su2_omp` writes the same file formats, with
English keys in its own log lines.

## Glossary of the French strings in the GPU programs

| string | meaning |
|---|---|
| `ERREUR` | error |
| `argument inconnu` | unknown argument |
| `L doit être pair >= 2` | L must be even and at least 2 |
| `beta invalide` | invalid beta |
| `voir en-tête de su2_hip.cpp` | see the header of su2_hip.cpp |
| `ERREUR : plafond d'essais atteint N fois` | error: heat-bath trial cap reached N times |
| `ERREUR historique plein` | internal history buffer full |
| `SELFTEST ... bornes` / `ECHEC` | bounds / failure |
| PERF line: `essais_par_lien_hb` | mean number of heat-bath trials per link update |
| PERF line: `plafond_atteint` | number of trial-cap events (must be 0) |
| PERF line: `wall_therm`, `wall`, `pas_par_s` | wall time of thermalisation, total wall time (s), steps per second |
| BENCH line: `hb_s_par_sweep`, `or_s_par_sweep` | seconds per heat-bath / overrelaxation sweep |
| BENCH line: `liens_par_s_hb`, `liens_par_s_or` | link updates per second |
| TESTHB line: `plafond` | number of trial-cap events |
| manifest `statut`: `EN_COURS` / `COMPLETE` | status: running / complete |
| manifest `rng`: `compteur (lien,sweep,tirage,etiquette), cle=graine` | counter (link, sweep, draw, tag), key = seed |
| manifest `step`: `1 sweep bain thermique ... damier 4 mu x 2 parites` | 1 heat-bath sweep ... checkerboard 4 mu × 2 parities |
| manifest `index`: `0 = etat initial ; 1..N apres skip_steps pas ; dump au meme indice` | 0 = initial state; 1..N after skip_steps steps; dumps at the same index |
| manifest `backend` (CUDA): `portage de su2_hip` | port of su2_hip |
| `non_fourni` | not provided |
