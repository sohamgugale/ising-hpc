// Statistical correctness test: compare Metropolis and Wolff averages on a 4x4
// periodic lattice against exact enumeration of all 2^16 configurations.
// Passes if every estimate lies within 5 standard errors (batch means).
#include <cmath>
#include <cstdio>
#include <vector>

#include "lattice.hpp"
#include "wolff.hpp"

using namespace ising;

struct Exact {
    double e, absm, m2;
};

static Exact enumerate(int L, double T) {
    const int N = L * L;
    double Z = 0, Se = 0, Sm = 0, Sm2 = 0;
    std::vector<int> s(N);
    for (long cfg = 0; cfg < (1L << N); ++cfg) {
        for (int i = 0; i < N; ++i) s[i] = (cfg >> i) & 1 ? 1 : -1;
        int E = 0, M = 0;
        for (int r = 0; r < L; ++r)
            for (int c = 0; c < L; ++c) {
                const int i = r * L + c;
                M += s[i];
                E -= s[i] * (s[r * L + (c + 1) % L] + s[((r + 1) % L) * L + c]);
            }
        const double w = std::exp(-E / T);
        Z += w;
        Se += w * E / N;
        Sm += w * std::abs(M) / N;
        Sm2 += w * (double(M) / N) * (double(M) / N);
    }
    return {Se / Z, Sm / Z, Sm2 / Z};
}

struct Estimate {
    double mean, err;
};

static Estimate batch_means(const std::vector<double>& x, int nb = 50) {
    const size_t per = x.size() / nb;
    std::vector<double> b(nb, 0.0);
    for (int k = 0; k < nb; ++k) {
        for (size_t i = 0; i < per; ++i) b[k] += x[k * per + i];
        b[k] /= per;
    }
    double mean = 0;
    for (double v : b) mean += v;
    mean /= nb;
    double var = 0;
    for (double v : b) var += (v - mean) * (v - mean);
    var /= (nb - 1);
    return {mean, std::sqrt(var / nb)};
}

static int check(const char* what, Estimate est, double exact) {
    const double z = std::fabs(est.mean - exact) / est.err;
    const bool ok = z < 5.0;
    std::printf("  %-28s estimate %.5f +/- %.5f  exact %.5f  (%.1f sigma) %s\n", what, est.mean, est.err, exact, z,
                ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    const int L = 4;
    int failures = 0;
    for (double T : {1.8, 2.269185314213022, 3.0}) {
        const Exact ex = enumerate(L, T);
        std::printf("T = %.4f\n", T);

        // Metropolis, both kernels (single owner, periodic ghosts)
        for (int kernel = 0; kernel < 2; ++kernel) {
            Lattice lat(L, 0, L);
            init_random(lat, 7);
            fill_periodic_ghosts(lat);
            const Acceptance acc(1.0 / T);
            const Thresholds thr(acc);
            std::vector<double> e, am, m2;
            const long therm = 2000, sweeps = 400000;
            for (long sweep = 0; sweep < therm + sweeps; ++sweep) {
                for (int parity = 0; parity < 2; ++parity) {
                    if (kernel == 0) half_sweep(lat, parity, sweep, 7, acc);
                    else half_sweep_blocked(lat, parity, sweep, 7, thr);
                    fill_periodic_ghosts(lat);
                }
                if (sweep >= therm) {
                    const Sums s = local_sums(lat);
                    const double m = double(s.M) / (L * L);
                    e.push_back(double(s.E) / (L * L));
                    am.push_back(std::fabs(m));
                    m2.push_back(m * m);
                }
            }
            const char* name = kernel == 0 ? "baseline" : "blocked ";
            char label[64];
            std::snprintf(label, sizeof label, "metropolis-%s <e>", name);
            failures += check(label, batch_means(e), ex.e);
            std::snprintf(label, sizeof label, "metropolis-%s <|m|>", name);
            failures += check(label, batch_means(am), ex.absm);
            std::snprintf(label, sizeof label, "metropolis-%s <m^2>", name);
            failures += check(label, batch_means(m2), ex.m2);
        }

        // Wolff
        Wolff w(L, T, 11);
        std::vector<double> e, am, m2;
        for (int k = 0; k < 2000; ++k) w.step();
        for (int k = 0; k < 400000; ++k) {
            w.step();
            const double m = w.magnetization();
            e.push_back(w.energy());
            am.push_back(std::fabs(m));
            m2.push_back(m * m);
        }
        failures += check("wolff <e>", batch_means(e), ex.e);
        failures += check("wolff <|m|>", batch_means(am), ex.absm);
        failures += check("wolff <m^2>", batch_means(m2), ex.m2);
    }
    std::printf("exact-enumeration tests: %s\n", failures ? "FAILED" : "passed");
    return failures ? 1 : 0;
}
