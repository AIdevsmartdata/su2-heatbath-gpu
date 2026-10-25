// su2_metropolis_ref.cpp — independent CPU reference for the SU(2) Wilson action in 4D.
//
// Written separately from src/su2_hip.cpp and deliberately different in every implementation
// choice, so that agreement between the two is a meaningful check:
//   * links stored as 2x2 complex matrices (std::complex<double>), site-major layout U[s*4+mu],
//     neighbour tables precomputed (the GPU code uses quaternions, direction-major, on-the-fly);
//   * Metropolis algorithm with N_hit hits per link (the GPU code uses an exact heat bath plus
//     overrelaxation), lexicographic sweep (the GPU code uses a checkerboard);
//   * random numbers from std::mt19937_64 (the GPU code uses counter-based Philox4x32-10).
// Only the physics is shared: S = beta * sum_p (1 - 1/2 Re Tr U_p), periodic L^4 lattice.
//
// Proposal: U' = R U with R = r0 + i r.sigma, r0 = sqrt(1 - eps^2), r = eps * n, n uniform on S^2
// (R and R^+ are equally likely, so the proposal is symmetric). eps is tuned during thermalisation
// towards 50 % acceptance and then frozen. Links are re-projected onto SU(2) after every sweep.
//
// Usage: su2_metropolis_ref --L 6 --beta 2.3 --seed 7 [--start hot|cold] --therm 500 --nmeas 4000
//                           [--hits 8] --out plaquette.csv
// Output CSV: sweep,plaquette,acceptance (one line per measurement sweep, after thermalisation).

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <chrono>

typedef std::complex<double> cd;
struct M { cd a, b, c, d; };   // [[a, b], [c, d]]

static inline M mul(const M& x, const M& y) {
  return M{x.a * y.a + x.b * y.c, x.a * y.b + x.b * y.d, x.c * y.a + x.d * y.c, x.c * y.b + x.d * y.d};
}
static inline M dag(const M& x) { return M{std::conj(x.a), std::conj(x.c), std::conj(x.b), std::conj(x.d)}; }
static inline M add(const M& x, const M& y) { return M{x.a + y.a, x.b + y.b, x.c + y.c, x.d + y.d}; }
static inline double retr(const M& x) { return x.a.real() + x.d.real(); }
static inline double retr_prod(const M& x, const M& y) {   // Re Tr(x y) without forming the product
  return (x.a * y.a + x.b * y.c + x.c * y.b + x.d * y.d).real();
}
static inline M su2(double x0, double x1, double x2, double x3) {   // x0 + i (x1 s1 + x2 s2 + x3 s3)
  return M{cd(x0, x3), cd(x2, x1), cd(-x2, x1), cd(x0, -x3)};
}
static inline void project(M& u) {   // back onto SU(2): [[a, b], [-b*, a*]] with |a|^2 + |b|^2 = 1
  double n = std::sqrt(std::norm(u.a) + std::norm(u.b));
  cd a = u.a / n, b = u.b / n;
  u = M{a, b, -std::conj(b), std::conj(a)};
}

struct Lattice {
  int L; long V;
  std::vector<M> U;              // U[s*4 + mu]
  std::vector<long> up, dn;      // up[s*4 + mu] = site s + e_mu
  M& link(long s, int mu) { return U[s * 4 + mu]; }
};

static void build(Lattice& lat, int L) {
  lat.L = L; lat.V = (long)L * L * L * L;
  lat.U.assign(4 * lat.V, su2(1, 0, 0, 0));
  lat.up.resize(4 * lat.V); lat.dn.resize(4 * lat.V);
  for (long s = 0; s < lat.V; ++s) {
    int x[4]; long t = s;
    for (int k = 3; k >= 0; --k) { x[k] = (int)(t % L); t /= L; }
    for (int mu = 0; mu < 4; ++mu) {
      int y[4], z[4];
      for (int k = 0; k < 4; ++k) { y[k] = x[k]; z[k] = x[k]; }
      y[mu] = (x[mu] + 1) % L; z[mu] = (x[mu] + L - 1) % L;
      long su = 0, sd = 0;
      for (int k = 0; k < 4; ++k) { su = su * L + y[k]; sd = sd * L + z[k]; }
      lat.up[s * 4 + mu] = su; lat.dn[s * 4 + mu] = sd;
    }
  }
}

// staple sum A with sum over plaquettes through U_mu(x) = Re Tr(U_mu(x) A)
static M staple(Lattice& lat, long s, int mu) {
  M A = M{0, 0, 0, 0};
  long xp = lat.up[s * 4 + mu];
  for (int nu = 0; nu < 4; ++nu) {
    if (nu == mu) continue;
    long xn = lat.up[s * 4 + nu], xm = lat.dn[s * 4 + nu], xpm = lat.dn[xp * 4 + nu];
    M t = mul(mul(lat.link(xp, nu), dag(lat.link(xn, mu))), dag(lat.link(s, nu)));
    M b = mul(mul(dag(lat.link(xpm, nu)), dag(lat.link(xm, mu))), lat.link(xm, nu));
    A = add(A, add(t, b));
  }
  return A;
}

static double plaquette(Lattice& lat) {
  double acc = 0;
  for (long s = 0; s < lat.V; ++s)
    for (int mu = 0; mu < 4; ++mu)
      for (int nu = mu + 1; nu < 4; ++nu) {
        M p = mul(mul(lat.link(s, mu), lat.link(lat.up[s * 4 + mu], nu)),
                  mul(dag(lat.link(lat.up[s * 4 + nu], mu)), dag(lat.link(s, nu))));
        acc += 0.5 * retr(p);
      }
  return acc / (6.0 * (double)lat.V);
}

int main(int argc, char** argv) {
  int L = 6, hits = 8; double beta = 2.3; uint64_t seed = 7; bool hot = true;
  long therm = 500, nmeas = 4000; std::string out;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto nxt = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", k.c_str()); exit(2); } return argv[++i]; };
    if (k == "--L") L = atoi(nxt().c_str());
    else if (k == "--beta") beta = atof(nxt().c_str());
    else if (k == "--seed") seed = strtoull(nxt().c_str(), nullptr, 10);
    else if (k == "--start") { std::string v = nxt(); hot = (v == "hot"); if (v != "hot" && v != "cold") exit(2); }
    else if (k == "--therm") therm = atol(nxt().c_str());
    else if (k == "--nmeas") nmeas = atol(nxt().c_str());
    else if (k == "--hits") hits = atoi(nxt().c_str());
    else if (k == "--out") out = nxt();
    else { fprintf(stderr, "unknown argument: %s\n", k.c_str()); return 2; }
  }
  if (L < 2 || !(beta > 0) || out.empty() || hits < 1) { fprintf(stderr, "need L >= 2, beta > 0, --out FILE, hits >= 1\n"); return 2; }
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> U01(0.0, 1.0);
  std::normal_distribution<double> N01(0.0, 1.0);
  Lattice lat; build(lat, L);
  if (hot)
    for (auto& u : lat.U) {
      double x0 = N01(rng), x1 = N01(rng), x2 = N01(rng), x3 = N01(rng);
      double n = std::sqrt(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3);
      u = su2(x0 / n, x1 / n, x2 / n, x3 / n);
    }
  double eps = 0.5;
  long acc = 0, prop = 0;
  auto sweep = [&]() {
    for (long s = 0; s < lat.V; ++s)
      for (int mu = 0; mu < 4; ++mu) {
        M A = staple(lat, s, mu);
        M& u = lat.link(s, mu);
        double old = retr_prod(u, A);
        for (int h = 0; h < hits; ++h) {
          double ct = 2.0 * U01(rng) - 1.0, ph = 2.0 * M_PI * U01(rng);
          double st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
          M R = su2(std::sqrt(1.0 - eps * eps), eps * st * std::cos(ph), eps * st * std::sin(ph), eps * ct);
          M un = mul(R, u);
          double neu = retr_prod(un, A);
          double dS = -0.5 * beta * (neu - old);
          ++prop;
          if (dS <= 0 || U01(rng) < std::exp(-dS)) { u = un; old = neu; ++acc; }
        }
      }
    for (auto& u : lat.U) project(u);
  };
  auto t0 = std::chrono::steady_clock::now();
  for (long t = 0; t < therm; ++t) {
    acc = prop = 0;
    sweep();
    double r = (double)acc / (double)prop;          // tune eps during thermalisation only
    if (t < therm * 3 / 4) eps = std::min(0.95, std::max(0.02, eps * (r > 0.5 ? 1.05 : 0.95)));
  }
  FILE* f = fopen(out.c_str(), "w");
  if (!f) { fprintf(stderr, "cannot open %s\n", out.c_str()); return 4; }
  fprintf(f, "sweep,plaquette,acceptance\n");
  double sum = 0;
  for (long m = 1; m <= nmeas; ++m) {
    acc = prop = 0;
    sweep();
    double p = plaquette(lat);
    sum += p;
    fprintf(f, "%ld,%.17g,%.6f\n", m, p, (double)acc / (double)prop);
  }
  fclose(f);
  double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("METROPOLIS_REF L=%d beta=%.17g seed=%llu hits=%d eps=%.4f therm=%ld nmeas=%ld mean_plaquette=%.10f wall=%.2f\n",
         L, beta, (unsigned long long)seed, hits, eps, therm, nmeas, sum / (double)nmeas, wall);
  return 0;
}
