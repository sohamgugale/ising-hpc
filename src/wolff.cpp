// Wolff cluster simulation; writes a time series for comparison with Metropolis.
// Time is reported in sweep-equivalents (cumulative flipped spins / N), the
// standard unit for comparing cluster and local algorithms.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "wolff.hpp"

int main(int argc, char** argv) {
    int L = 64;
    double T = 2.269185314213022;
    long clusters = 100000, therm = 1000, every = 1;
    std::uint64_t seed = 12345;
    std::string out;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) std::exit(2);
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--L")) L = std::atoi(next());
        else if (!std::strcmp(argv[i], "--T")) T = std::atof(next());
        else if (!std::strcmp(argv[i], "--clusters")) clusters = std::atol(next());
        else if (!std::strcmp(argv[i], "--therm")) therm = std::atol(next());
        else if (!std::strcmp(argv[i], "--seed")) seed = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(argv[i], "--out")) out = next();
        else if (!std::strcmp(argv[i], "--every")) every = std::atol(next());
        else {
            std::fprintf(stderr, "usage: %s [--L n] [--T temp] [--clusters n] [--therm n] [--seed n] [--every n] [--out file.csv]\n",
                         argv[0]);
            return 2;
        }
    }
    ising::Wolff w(L, T, seed);
    for (long k = 0; k < therm; ++k) w.step();

    std::FILE* f = out.empty() ? nullptr : std::fopen(out.c_str(), "w");
    if (f) std::fprintf(f, "step,sweeps,cluster,e,m\n");
    double flipped = 0.0, mean_cluster = 0.0;
    for (long k = 0; k < clusters; ++k) {
        const int c = w.step();
        flipped += c;
        mean_cluster += c;
        if (f && k % every == 0) std::fprintf(f, "%ld,%.6f,%d,%.10f,%.10f\n", k, flipped / w.sites(), c, w.energy(), w.magnetization());
    }
    if (f) std::fclose(f);
    std::printf("algorithm=wolff L=%d T=%.6f clusters=%ld mean_cluster_fraction=%.5f sweep_equivalents=%.1f\n", L, T,
                clusters, mean_cluster / clusters / w.sites(), flipped / w.sites());
    return 0;
}
