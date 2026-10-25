// su2_omp.cpp — CPU (OpenMP) transcription of the SAME algorithm as src/su2_hip.cpp.
//
// Purpose: an honest CPU baseline for throughput comparisons, and a CPU-only way to exercise the
// algorithm. Same action, same quaternion arithmetic, same exact heat-bath samplers (Creutz /
// Kennedy-Pendleton), same overrelaxation, same checkerboard (4 directions x 2 parities) and the
// same counter-based Philox4x32-10 streams (key = seed, counter = (link, sweep, draw, tag)).
// Consequences:
//   * the chain does not depend on the number of OpenMP threads (bit-identical output);
//   * the initial hot configuration equals the GPU one up to libm rounding (~1e-16); afterwards
//     the chains decorrelate chaotically, exactly like two different GPUs do (different libm).
// This is a plain transcription compiled with -O3 -march=native -fopenmp: no hand-written SIMD,
// no data-layout tuning. A tuned CPU code may well be faster; treat the numbers as a baseline.
//
// Output files have the same format as su2_hip (therm.csv, measurements.csv, cfg_XXXXXX.npy,
// manifest.json with "engine":"su2_omp"). Usage mirrors su2_hip:
//   su2_omp --L 6 --beta 2.5 --seed 1 --start hot|cold --therm 300 --nmeas 100 --skip 1 --nor 4
//           --dump-every 50 --outdir DIR [--dump-initial] [--source-hash H] [--threads N]
//   su2_omp --selftest
//   su2_omp --bench --L 16 --beta 2.5 --nsweeps 50 [--threads N]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <sys/stat.h>
#include <omp.h>

typedef uint32_t u32;
typedef uint64_t u64;

struct U4 { u32 v[4]; };

static inline U4 philox4x32_10(U4 c, u32 k0, u32 k1) {
  const u32 M0 = 0xD2511F53u, M1 = 0xCD9E8D57u, W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
  for (int r = 0; r < 10; ++r) {
    u64 p0 = (u64)M0 * (u64)c.v[0];
    u64 p1 = (u64)M1 * (u64)c.v[2];
    u32 hi0 = (u32)(p0 >> 32), lo0 = (u32)p0;
    u32 hi1 = (u32)(p1 >> 32), lo1 = (u32)p1;
    U4 n;
    n.v[0] = hi1 ^ c.v[1] ^ k0;
    n.v[1] = lo1;
    n.v[2] = hi0 ^ c.v[3] ^ k1;
    n.v[3] = lo0;
    c = n;
    k0 += W0; k1 += W1;
  }
  return c;
}

static inline double u01(u32 a, u32 b) {
  u64 m = ((u64)(a >> 6) << 26) | (u64)(b >> 6);
  return ((double)m + 0.5) * (1.0 / 4503599627370496.0);
}

enum : u32 { TAG_INIT = 0x494E4954u, TAG_HB = 0x48424154u };

struct Q { double x, y, z, w; };
static inline Q qmul(Q a, Q b) {
  Q c;
  c.x = a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w;
  c.y = a.x * b.y + b.x * a.y - (a.z * b.w - a.w * b.z);
  c.z = a.x * b.z + b.x * a.z - (a.w * b.y - a.y * b.w);
  c.w = a.x * b.w + b.x * a.w - (a.y * b.z - a.z * b.y);
  return c;
}
static inline Q qdag(Q a) { Q c = {a.x, -a.y, -a.z, -a.w}; return c; }
static inline Q qadd(Q a, Q b) { Q c = {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; return c; }
static inline double qnorm(Q a) { return sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w); }
static inline Q qscale(Q a, double s) { Q c = {a.x * s, a.y * s, a.z * s, a.w * s}; return c; }

struct Geo { int L; long V; long st[4]; };
static inline long nb_up(const Geo& g, long s, int d, int xd) { return (xd == g.L - 1) ? s - (long)(g.L - 1) * g.st[d] : s + g.st[d]; }
static inline long nb_dn(const Geo& g, long s, int d, int xd) { return (xd == 0) ? s + (long)(g.L - 1) * g.st[d] : s - g.st[d]; }
static inline void coords(const Geo& g, long s, int x[4]) {
  x[3] = (int)(s % g.L); long t = s / g.L;
  x[2] = (int)(t % g.L); t /= g.L;
  x[1] = (int)(t % g.L); x[0] = (int)(t / g.L);
}
static inline Q ld(const Q* U, const Geo& g, int mu, long s) { return U[(long)mu * g.V + s]; }

static inline Q staple(const Q* U, const Geo& g, int mu, long s, const int x[4]) {
  Q A = {0, 0, 0, 0};
  long smu = nb_up(g, s, mu, x[mu]);
  for (int nu = 0; nu < 4; ++nu) {
    if (nu == mu) continue;
    long snu = nb_up(g, s, nu, x[nu]);
    long sdn = nb_dn(g, s, nu, x[nu]);
    long smu_dn = nb_dn(g, smu, nu, x[nu]);
    Q t = qmul(qmul(ld(U, g, nu, smu), qdag(ld(U, g, mu, snu))), qdag(ld(U, g, nu, s)));
    Q b = qmul(qmul(qdag(ld(U, g, nu, smu_dn)), qdag(ld(U, g, mu, sdn))), ld(U, g, nu, sdn));
    A = qadd(A, qadd(t, b));
  }
  return A;
}

static inline long site_of(const Geo& g, long t, int parity, int x[4]) {
  long s = 2 * t;
  coords(g, s, x);
  int p = (x[0] + x[1] + x[2] + x[3]) & 1;
  if (p != parity) { s += 1; x[3] += 1; }
  return s;
}

// Haar-random SU(2) element from two Philox blocks (Box-Muller on 4 Gaussians, then normalisation)
static inline Q haar_from(U4 r1, U4 r2) {
  double u1 = u01(r1.v[0], r1.v[1]), u2 = u01(r1.v[2], r1.v[3]);
  double u3 = u01(r2.v[0], r2.v[1]), u4 = u01(r2.v[2], r2.v[3]);
  double ra = sqrt(-2.0 * log(u1)), rb = sqrt(-2.0 * log(u3));
  Q q = {ra * cos(2 * M_PI * u2), ra * sin(2 * M_PI * u2), rb * cos(2 * M_PI * u4), rb * sin(2 * M_PI * u4)};
  return qscale(q, 1.0 / qnorm(q));
}

static inline double sample_x0(double alpha, u32 li, u32 sweep_id, u32& draw, u32 k0, u32 k1, u32& ntry, bool& ok) {
  const u32 CAP = 1000000u;
  double x0 = 1.0; ok = false; ntry = 0;
  if (alpha >= 2.0) {
    while (ntry < CAP) {
      U4 r1 = philox4x32_10(U4{{li, sweep_id, draw++, TAG_HB}}, k0, k1);
      U4 r2 = philox4x32_10(U4{{li, sweep_id, draw++, TAG_HB}}, k0, k1);
      double a1 = u01(r1.v[0], r1.v[1]), a2 = u01(r1.v[2], r1.v[3]);
      double a3 = u01(r2.v[0], r2.v[1]), a4 = u01(r2.v[2], r2.v[3]);
      double c = cos(2 * M_PI * a2);
      double lam2 = -(log(a1) + c * c * log(a3)) / (2.0 * alpha);
      ++ntry;
      if (a4 * a4 <= 1.0 - lam2) { x0 = 1.0 - 2.0 * lam2; ok = true; break; }
    }
  } else {
    double e2 = exp(-2.0 * alpha);
    while (ntry < CAP) {
      U4 r1 = philox4x32_10(U4{{li, sweep_id, draw++, TAG_HB}}, k0, k1);
      double a1 = u01(r1.v[0], r1.v[1]), a2 = u01(r1.v[2], r1.v[3]);
      double y = 1.0 + log(e2 + a1 * (1.0 - e2)) / alpha;
      ++ntry;
      if (a2 * a2 <= 1.0 - y * y) { x0 = y; ok = true; break; }
    }
  }
  return x0;
}

struct Lat {
  Geo g; std::vector<Q> U; u32 k0, k1; u32 sweep_id = 0;
  unsigned long long caps = 0, tries = 0; long n_hb = 0, n_or = 0;
  std::vector<double> site;   // per-site scratch (plaquette)
};

static void lat_init(Lat& l, int L, u64 seed, bool hot) {
  l.g.L = L; l.g.V = (long)L * L * L * L;
  l.g.st[3] = 1; l.g.st[2] = L; l.g.st[1] = (long)L * L; l.g.st[0] = (long)L * L * L;
  l.k0 = (u32)(seed & 0xFFFFFFFFu); l.k1 = (u32)(seed >> 32);
  l.U.resize(4 * l.g.V); l.site.resize(l.g.V);
  long n = 4 * l.g.V;
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < n; ++i) {
    if (!hot) { l.U[i] = Q{1, 0, 0, 0}; continue; }
    U4 r1 = philox4x32_10(U4{{(u32)i, 0xFFFFFFFFu, 0u, TAG_INIT}}, l.k0, l.k1);
    U4 r2 = philox4x32_10(U4{{(u32)i, 0xFFFFFFFFu, 1u, TAG_INIT}}, l.k0, l.k1);
    l.U[i] = haar_from(r1, r2);
  }
}

static void sweep_hb(Lat& l, double beta) {
  const Geo g = l.g; Q* U = l.U.data(); const u32 k0 = l.k0, k1 = l.k1, sw = l.sweep_id;
  unsigned long long caps = 0, tries = 0;
  for (int mu = 0; mu < 4; ++mu)
    for (int p = 0; p < 2; ++p) {
      #pragma omp parallel for schedule(static) reduction(+:caps,tries)
      for (long t = 0; t < g.V / 2; ++t) {
        int x[4];
        long s = site_of(g, t, p, x);
        long li = (long)mu * g.V + s;
        Q A = staple(U, g, mu, s, x);
        double k = qnorm(A);
        u32 draw = 0;
        if (!(k > 1e-300)) {   // zero staple: Haar measure (same counters as the GPU code)
          U4 r1 = philox4x32_10(U4{{(u32)li, sw, draw++, TAG_HB}}, k0, k1);
          U4 r2 = philox4x32_10(U4{{(u32)li, sw, draw++, TAG_HB}}, k0, k1);
          U[li] = haar_from(r1, r2);
          continue;
        }
        Q Vv = qscale(A, 1.0 / k);
        double alpha = beta * k;
        u32 ntry = 0; bool ok = false;
        double x0 = sample_x0(alpha, (u32)li, sw, draw, k0, k1, ntry, ok);
        tries += ntry;
        if (!ok) { caps += 1; continue; }
        U4 rd = philox4x32_10(U4{{(u32)li, sw, draw++, TAG_HB}}, k0, k1);
        double ct = 2.0 * u01(rd.v[0], rd.v[1]) - 1.0;
        double ph = 2 * M_PI * u01(rd.v[2], rd.v[3]);
        double st = sqrt(fmax(0.0, 1.0 - ct * ct));
        double rr = sqrt(fmax(0.0, 1.0 - x0 * x0));
        Q W; W.x = x0; W.y = rr * st * cos(ph); W.z = rr * st * sin(ph); W.w = rr * ct;
        Q Un = qmul(W, qdag(Vv));
        U[li] = qscale(Un, 1.0 / qnorm(Un));
      }
    }
  l.caps += caps; l.tries += tries; l.n_hb++; l.sweep_id++;
}

static void sweep_or(Lat& l) {
  const Geo g = l.g; Q* U = l.U.data();
  for (int mu = 0; mu < 4; ++mu)
    for (int p = 0; p < 2; ++p) {
      #pragma omp parallel for schedule(static)
      for (long t = 0; t < g.V / 2; ++t) {
        int x[4];
        long s = site_of(g, t, p, x);
        long li = (long)mu * g.V + s;
        Q A = staple(U, g, mu, s, x);
        double k = qnorm(A);
        if (!(k > 1e-300)) continue;
        Q Vd = qdag(qscale(A, 1.0 / k));
        Q Un = qmul(qmul(Vd, qdag(U[li])), Vd);
        U[li] = qscale(Un, 1.0 / qnorm(Un));
      }
    }
  l.n_or++;
}

// plaquette: per-site values in parallel, then a sequential sum (independent of the thread count)
static double plaquette(Lat& l) {
  const Geo g = l.g; const Q* U = l.U.data(); double* out = l.site.data();
  #pragma omp parallel for schedule(static)
  for (long s = 0; s < g.V; ++s) {
    int x[4]; coords(g, s, x);
    double acc = 0;
    for (int mu = 0; mu < 4; ++mu) {
      long smu = nb_up(g, s, mu, x[mu]);
      for (int nu = mu + 1; nu < 4; ++nu) {
        long snu = nb_up(g, s, nu, x[nu]);
        Q p = qmul(qmul(ld(U, g, mu, s), ld(U, g, nu, smu)), qmul(qdag(ld(U, g, mu, snu)), qdag(ld(U, g, nu, s))));
        acc += p.x;
      }
    }
    out[s] = acc;
  }
  double sum = 0; for (long s = 0; s < g.V; ++s) sum += out[s];
  return sum / (6.0 * (double)g.V);
}

static void polyakov(Lat& l, double P[4]) {
  const Geo g = l.g; const Q* U = l.U.data(); long L3 = g.V / g.L;
  std::vector<double> v(4 * L3);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < 4 * L3; ++i) {
    int mu = (int)(i / L3); long j = i % L3;
    int x[4]; x[mu] = 0; int dirs[3], nd = 0;
    for (int d = 0; d < 4; ++d) if (d != mu) dirs[nd++] = d;
    x[dirs[2]] = (int)(j % g.L); long t = j / g.L;
    x[dirs[1]] = (int)(t % g.L); x[dirs[0]] = (int)(t / g.L);
    long s = ((long)(x[0] * g.L + x[1]) * g.L + x[2]) * g.L + x[3];
    Q Pq = {1, 0, 0, 0};
    for (int k = 0; k < g.L; ++k) { Pq = qmul(Pq, ld(U, g, mu, s)); s += (k == g.L - 1) ? 0 : g.st[mu]; }
    v[i] = Pq.x;
  }
  for (int mu = 0; mu < 4; ++mu) { double s = 0; for (long j = 0; j < L3; ++j) s += v[mu * L3 + j]; P[mu] = s / (double)L3; }
}

static bool dump_npy(const Lat& l, const std::string& path) {
  long n = 4 * l.g.V; int L = l.g.L;
  char descr[256];
  snprintf(descr, sizeof descr, "{'descr': '<c16', 'fortran_order': False, 'shape': (4, %d, %d, %d, %d, 2, 2), }", L, L, L, L);
  std::string hd = descr;
  size_t tot = 10 + hd.size() + 1, pad = (64 - tot % 64) % 64;
  hd += std::string(pad, ' '); hd += "\n";
  std::string tmp = path + ".part";
  FILE* f = fopen(tmp.c_str(), "wb");
  if (!f) { fprintf(stderr, "ERROR opening %s\n", tmp.c_str()); return false; }
  const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  fwrite(magic, 1, 8, f);
  uint16_t hl = (uint16_t)hd.size(); fwrite(&hl, 2, 1, f);
  fwrite(hd.data(), 1, hd.size(), f);
  size_t nw = 0;
  for (long i = 0; i < n; ++i) {
    const Q& q = l.U[i];
    double m[8] = {q.x, q.w, q.z, q.y, -q.z, q.y, q.x, -q.w};
    nw += fwrite(m, 8, 8, f);
  }
  bool ok = (nw == (size_t)(8 * n)) && (fclose(f) == 0);
  if (!ok) { fprintf(stderr, "ERROR writing %s\n", tmp.c_str()); return false; }
  return rename(tmp.c_str(), path.c_str()) == 0;
}

struct Args {
  int L = 8; double beta = 2.5; u64 seed = 1; bool hot = true;
  long therm = 1000, nmeas = 100, skip = 1; int nor = 4; long dump_every = 0;
  std::string outdir = "."; bool dump_initial = false; std::string source_hash = "not_provided";
  bool selftest = false, bench = false; long nsweeps = 50; int threads = 0;
};

static void usage() { fprintf(stderr, "see the header of cpu/su2_omp.cpp\n"); exit(2); }

static Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto nxt = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
    if (k == "--L") a.L = atoi(nxt().c_str());
    else if (k == "--beta") a.beta = atof(nxt().c_str());
    else if (k == "--seed") a.seed = strtoull(nxt().c_str(), nullptr, 10);
    else if (k == "--start") { std::string v = nxt(); if (v == "hot") a.hot = true; else if (v == "cold") a.hot = false; else usage(); }
    else if (k == "--therm") a.therm = atol(nxt().c_str());
    else if (k == "--nmeas") a.nmeas = atol(nxt().c_str());
    else if (k == "--skip") a.skip = atol(nxt().c_str());
    else if (k == "--nor") a.nor = atoi(nxt().c_str());
    else if (k == "--dump-every") a.dump_every = atol(nxt().c_str());
    else if (k == "--outdir") a.outdir = nxt();
    else if (k == "--dump-initial") a.dump_initial = true;
    else if (k == "--source-hash") a.source_hash = nxt();
    else if (k == "--selftest") a.selftest = true;
    else if (k == "--bench") a.bench = true;
    else if (k == "--nsweeps") a.nsweeps = atol(nxt().c_str());
    else if (k == "--threads") a.threads = atoi(nxt().c_str());
    else { fprintf(stderr, "unknown argument: %s\n", k.c_str()); usage(); }
  }
  if (a.L < 2 || (a.L % 2) != 0) { fprintf(stderr, "L must be even and >= 2\n"); exit(2); }
  if (!(a.beta > 0) || !std::isfinite(a.beta)) { fprintf(stderr, "beta must be finite and > 0\n"); exit(2); }
  if (a.skip < 1 || a.nor < 0) { fprintf(stderr, "skip >= 1, nor >= 0\n"); exit(2); }
  if (4.0 * pow((double)a.L, 4) >= 4294967296.0) { fprintf(stderr, "4 L^4 must be < 2^32 (32-bit link counter)\n"); exit(2); }
  return a;
}

static int selftest() {
  int bad = 0;
  struct { u32 c[4], k[2], r[4]; } kat[3] = {
    {{0, 0, 0, 0}, {0, 0}, {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}},
    {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, {0xffffffffu, 0xffffffffu}, {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}},
    {{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u}, {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}}};
  for (int t = 0; t < 3; ++t) {
    U4 c = {{kat[t].c[0], kat[t].c[1], kat[t].c[2], kat[t].c[3]}};
    U4 r = philox4x32_10(c, kat[t].k[0], kat[t].k[1]);
    bool ok = true; for (int j = 0; j < 4; ++j) ok &= (r.v[j] == kat[t].r[j]);
    printf("SELFTEST philox KAT %d : %s\n", t, ok ? "PASS" : "FAIL");
    bad += !ok;
  }
  double lo = u01(0, 0), hi = u01(0xffffffffu, 0xffffffffu);
  bool okb = lo > 0 && hi < 1;
  printf("SELFTEST u01 bounds : %s\n", okb ? "PASS" : "FAIL");
  bad += !okb;
  return bad;
}

static void write_manifest(const Args& a, const std::string& path, const char* status, int nthreads) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) { fprintf(stderr, "ERROR manifest\n"); exit(4); }
  fprintf(f, "{\n\"schema\":1,\"engine\":\"su2_omp\",\"source_hash\":\"%s\",\"backend\":\"CPU OpenMP FP64\",\"threads\":%d,"
             "\"rng\":\"Philox4x32-10 counter (link,sweep,draw,tag), key=seed\",\"l\":%d,\"beta\":%.17g,"
             "\"seed\":%llu,\"start\":\"%s\",\"therm_steps\":%ld,\"measurements\":%ld,\"skip_steps\":%ld,\"or_per_step\":%d,"
             "\"dump_every\":%ld,\"action\":\"beta sum(1-ReTr(U_p)/2)\","
             "\"step\":\"1 heat-bath sweep (Kennedy-Pendleton alpha>=2, Creutz alpha<2) + or_per_step OR sweeps, checkerboard 4 mu x 2 parities\","
             "\"layout\":\"complex128 C-order (mu,x0,x1,x2,x3,i,j), temporal=3\","
             "\"index\":\"0 = initial state ; 1..N after skip_steps steps ; dump at the same index\",\"statut\":\"%s\"}\n",
          a.source_hash.c_str(), nthreads, a.L, a.beta, (unsigned long long)a.seed, a.hot ? "Hot" : "Cold", a.therm, a.nmeas, a.skip,
          a.nor, a.dump_every, status);
  fclose(f);
}

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  if (a.selftest) { int b = selftest(); printf("SELFTEST %s\n", b ? "FAILED" : "OK"); return b ? 1 : 0; }
  if (a.threads > 0) omp_set_num_threads(a.threads);
  int nth = omp_get_max_threads();
  fprintf(stderr, "# su2_omp : %d OpenMP threads\n", nth);
  Lat l; lat_init(l, a.L, a.seed, a.hot);
  if (a.bench) {
    for (int i = 0; i < 2; ++i) { sweep_hb(l, a.beta); for (int k = 0; k < 4; ++k) sweep_or(l); }   // warm-up
    l.n_hb = l.n_or = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < a.nsweeps; ++i) sweep_hb(l, a.beta);
    auto t1 = std::chrono::steady_clock::now();
    for (long i = 0; i < a.nsweeps; ++i) sweep_or(l);
    auto t2 = std::chrono::steady_clock::now();
    double P = plaquette(l);
    double thb = std::chrono::duration<double>(t1 - t0).count(), tor = std::chrono::duration<double>(t2 - t1).count();
    double nl = 4.0 * l.g.V;
    printf("BENCH_CPU L=%d beta=%g V=%ld threads=%d hb_sweeps=%ld hb_s_per_sweep=%.6e or_sweeps=%ld or_s_per_sweep=%.6e "
           "links_per_s_hb=%.4e links_per_s_or=%.4e P=%.10f caps=%llu\n",
           a.L, a.beta, l.g.V, nth, l.n_hb, thb / l.n_hb, l.n_or, tor / l.n_or, nl * l.n_hb / thb, nl * l.n_or / tor, P, l.caps);
    return l.caps ? 6 : 0;
  }
  mkdir(a.outdir.c_str(), 0755);
  write_manifest(a, a.outdir + "/manifest.json", "EN_COURS", nth);
  FILE* ft = fopen((a.outdir + "/therm.csv").c_str(), "w");
  FILE* fm = fopen((a.outdir + "/measurements.csv").c_str(), "w");
  if (!ft || !fm) { fprintf(stderr, "ERROR csv\n"); return 4; }
  fprintf(ft, "step,plaquette,P0,P1,P2,P3\n");
  fprintf(fm, "measurement_index,total_steps,phase,plaquette,P0,P1,P2,P3\n");
  auto meas = [&](double r[5]) { r[0] = plaquette(l); polyakov(l, r + 1); };
  auto step = [&]() { sweep_hb(l, a.beta); for (int i = 0; i < a.nor; ++i) sweep_or(l); };
  auto w0 = std::chrono::steady_clock::now();
  double r[5];
  meas(r);
  fprintf(ft, "%d,%.17g,%.17g,%.17g,%.17g,%.17g\n", 0, r[0], r[1], r[2], r[3], r[4]);
  if (a.dump_initial && !dump_npy(l, a.outdir + "/cfg_000000.npy")) return 5;
  for (long t = 1; t <= a.therm; ++t) {
    step(); meas(r);
    fprintf(ft, "%ld,%.17g,%.17g,%.17g,%.17g,%.17g\n", t, r[0], r[1], r[2], r[3], r[4]);
  }
  fclose(ft);
  meas(r);
  fprintf(fm, "%d,%ld,%s,%.17g,%.17g,%.17g,%.17g,%.17g\n", 0, a.therm, "initial", r[0], r[1], r[2], r[3], r[4]);
  int ndump = 0;
  for (long m = 1; m <= a.nmeas; ++m) {
    for (long k = 0; k < a.skip; ++k) step();
    meas(r);
    fprintf(fm, "%ld,%ld,%s,%.17g,%.17g,%.17g,%.17g,%.17g\n", m, a.therm + m * a.skip, "sample", r[0], r[1], r[2], r[3], r[4]);
    if (a.dump_every > 0 && m % a.dump_every == 0) {
      char nm[64]; snprintf(nm, sizeof nm, "/cfg_%06ld.npy", m);
      if (!dump_npy(l, a.outdir + nm)) return 5;
      ++ndump;
    }
  }
  fclose(fm);
  double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
  double hb_updates = 4.0 * l.g.V * (double)l.n_hb;
  printf("PERF_CPU L=%d beta=%g threads=%d hb_sweeps=%ld or_sweeps=%ld trials_per_link_hb=%.5f caps=%llu dumps=%d wall=%.2f steps_per_s=%.2f\n",
         a.L, a.beta, nth, l.n_hb, l.n_or, (double)l.tries / fmax(hb_updates, 1.0), l.caps, ndump, wall,
         (double)(a.therm + a.nmeas * a.skip) / fmax(wall, 1e-9));
  if (l.caps != 0) { fprintf(stderr, "ERROR: trial cap reached %llu times\n", l.caps); return 6; }
  write_manifest(a, a.outdir + "/manifest.json", "COMPLETE", nth);
  FILE* fc = fopen((a.outdir + "/COMPLETE").c_str(), "w"); if (fc) fclose(fc);
  return 0;
}
