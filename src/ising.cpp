// Checkerboard Metropolis simulation of the 2D Ising model.
//
// Built twice by CMake:
//   ising      - one process, OpenMP threads over rows
//   ising_mpi  - row-strip domain decomposition over MPI ranks, halo exchange
//                of one ghost row per side after every half-sweep
//
// Both print one summary line (key=value pairs) and can write a time series of
// per-site energy and magnetisation for analysis.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef ISING_MPI
#include <mpi.h>
#endif

#include "lattice.hpp"

using namespace ising;

struct Options {
    int L = 64;
    double T = 2.269185314213022;  // exact critical temperature 2 / ln(1 + sqrt 2)
    long sweeps = 10000;
    long therm = 1000;
    long every = 1;  // measure every `every` sweeps; 0 disables measurements
    std::uint64_t seed = 12345;
    std::string out;  // time-series CSV (rank 0 writes it)
    std::string start = "hot";  // "hot" (random) or "cold" (all spins up)
    std::string kernel = "batched";  // "batched" (fastest), "blocked" or "baseline"
};

static Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--L")) o.L = std::atoi(next("--L"));
        else if (!std::strcmp(argv[i], "--T")) o.T = std::atof(next("--T"));
        else if (!std::strcmp(argv[i], "--sweeps")) o.sweeps = std::atol(next("--sweeps"));
        else if (!std::strcmp(argv[i], "--therm")) o.therm = std::atol(next("--therm"));
        else if (!std::strcmp(argv[i], "--every")) o.every = std::atol(next("--every"));
        else if (!std::strcmp(argv[i], "--seed")) o.seed = std::strtoull(next("--seed"), nullptr, 10);
        else if (!std::strcmp(argv[i], "--out")) o.out = next("--out");
        else if (!std::strcmp(argv[i], "--kernel")) o.kernel = next("--kernel");
        else if (!std::strcmp(argv[i], "--start")) o.start = next("--start");
        else {
            std::fprintf(stderr,
                         "usage: %s [--L n] [--T temp] [--sweeps n] [--therm n] [--every n] [--seed n] [--out file.csv] [--kernel batched|blocked|baseline] [--start hot|cold]\n",
                         argv[0]);
            std::exit(2);
        }
    }
    if (o.kernel != "batched" && o.kernel != "blocked" && o.kernel != "baseline") {
        std::fprintf(stderr, "--kernel must be batched, blocked or baseline\n");
        std::exit(2);
    }
    if (o.start != "hot" && o.start != "cold") {
        std::fprintf(stderr, "--start must be hot or cold\n");
        std::exit(2);
    }
    if (o.L < 4 || o.L % 2) {
        std::fprintf(stderr, "L must be even and >= 4\n");
        std::exit(2);
    }
    return o;
}

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    int rank = 0, nranks = 1;
#ifdef ISING_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif
    const Options o = parse(argc, argv);
    if (o.L < nranks) {
        if (rank == 0) std::fprintf(stderr, "need at least one row per rank\n");
#ifdef ISING_MPI
        MPI_Abort(MPI_COMM_WORLD, 2);
#endif
        return 2;
    }

    // Row-strip decomposition: the first L % nranks ranks get one extra row.
    const int base = o.L / nranks, extra = o.L % nranks;
    const int nrows = base + (rank < extra ? 1 : 0);
    const int row0 = rank * base + (rank < extra ? rank : extra);
    Lattice lat(o.L, row0, nrows);
    if (o.start == "hot") init_random(lat, o.seed);  // Lattice() already sets every spin to +1 (cold start)
    const Acceptance acc(1.0 / o.T);
    const Thresholds thr(acc);
    const int kernel = o.kernel == "batched" ? 2 : o.kernel == "blocked" ? 1 : 0;
    auto update = [&](int parity, long sweep) {
        const auto sw = static_cast<std::uint64_t>(sweep);
        if (kernel == 2) half_sweep_batched(lat, parity, sw, o.seed, thr);
        else if (kernel == 1) half_sweep_blocked(lat, parity, sw, o.seed, thr);
        else half_sweep(lat, parity, sw, o.seed, acc);
    };

#ifdef ISING_MPI
    const int up = (rank - 1 + nranks) % nranks, down = (rank + 1) % nranks;
    auto exchange = [&]() {
        // last owned row -> down neighbour's top ghost; first owned row -> up neighbour's bottom ghost
        MPI_Sendrecv(lat.row(nrows), o.L, MPI_INT8_T, down, 0, lat.row(0), o.L, MPI_INT8_T, up, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        MPI_Sendrecv(lat.row(1), o.L, MPI_INT8_T, up, 1, lat.row(nrows + 1), o.L, MPI_INT8_T, down, 1, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
    };
#else
    auto exchange = [&]() { fill_periodic_ghosts(lat); };
#endif

    std::FILE* ts = nullptr;
    if (rank == 0 && !o.out.empty() && o.every > 0) {
        ts = std::fopen(o.out.c_str(), "w");
        if (!ts) {
            std::perror(o.out.c_str());
            return 1;
        }
        std::fprintf(ts, "sweep,e,m\n");
    }

    const double N = static_cast<double>(o.L) * o.L;
    exchange();
    double t_update = 0.0;
    const long total = o.therm + o.sweeps;
    for (long sweep = 0; sweep < total; ++sweep) {
        const double t0 = now();
        update(0, sweep);
        exchange();
        update(1, sweep);
        exchange();
        t_update += now() - t0;

        if (o.every > 0 && sweep >= o.therm && (sweep - o.therm) % o.every == 0) {
            const Sums loc = local_sums(lat);
            long long glob[2] = {loc.E, loc.M};
#ifdef ISING_MPI
            MPI_Allreduce(MPI_IN_PLACE, glob, 2, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
            if (ts) std::fprintf(ts, "%ld,%.10f,%.10f\n", sweep - o.therm, glob[0] / N, glob[1] / N);
        }
    }
    if (ts) std::fclose(ts);

    // Checksum of the final configuration in global row order.
    std::uint64_t checksum = 0;
#ifdef ISING_MPI
    double t_max = 0.0;
    MPI_Reduce(&t_update, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    t_update = t_max;
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        counts[r] = (base + (r < extra ? 1 : 0)) * o.L;
        displs[r] = (r * base + (r < extra ? r : extra)) * o.L;
    }
    std::vector<std::int8_t> all(rank == 0 ? static_cast<size_t>(o.L) * o.L : 0);
    MPI_Gatherv(lat.row(1), nrows * o.L, MPI_INT8_T, all.data(), counts.data(), displs.data(), MPI_INT8_T, 0,
                MPI_COMM_WORLD);
    if (rank == 0) checksum = fnv1a(all.data(), all.size());
#else
    checksum = fnv1a(lat.row(1), static_cast<size_t>(nrows) * o.L);
#endif

    if (rank == 0) {
        int threads = 1;
#ifdef _OPENMP
        threads = omp_get_max_threads();
#endif
        const double flips_per_ns = N * static_cast<double>(total) / (t_update * 1e9);
#ifdef ISING_MPI
        const char* backend = "mpi";
#elif defined(_OPENMP)
        const char* backend = "openmp";
#else
        const char* backend = "serial";
#endif
        std::printf("backend=%s kernel=%s ranks=%d threads=%d L=%d T=%.6f sweeps=%ld time_s=%.4f flips_per_ns=%.4f checksum=%016llx\n",
                    backend, o.kernel.c_str(), nranks, threads, o.L, o.T, total, t_update, flips_per_ns,
                    static_cast<unsigned long long>(checksum));
    }
#ifdef ISING_MPI
    MPI_Finalize();
#endif
    return 0;
}
