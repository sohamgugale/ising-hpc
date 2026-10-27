// 2D Ising model on an L x L periodic lattice, checkerboard Metropolis updates.
//
// A rank owns `nrows` consecutive global rows starting at `row0`, stored with
// one ghost row above (index 0) and one below (index nrows + 1). The serial and
// OpenMP backends are the special case of a single owner whose ghost rows are
// periodic copies of its own boundary rows.
//
// Sites with (row + col) even form sublattice 0, the rest sublattice 1. All
// neighbours of a site lie on the other sublattice, so every site of one
// colour can be updated simultaneously without data races.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "philox.hpp"

namespace ising {

struct Lattice {
    int L = 0;      // global linear size (even)
    int row0 = 0;   // first global row owned
    int nrows = 0;  // rows owned
    std::vector<std::int8_t> s;  // (nrows + 2) * L, ghost rows at 0 and nrows + 1

    Lattice(int L_, int row0_, int nrows_) : L(L_), row0(row0_), nrows(nrows_), s(static_cast<size_t>(nrows_ + 2) * L_, 1) {}

    std::int8_t* row(int r) { return s.data() + static_cast<size_t>(r) * L; }
    const std::int8_t* row(int r) const { return s.data() + static_cast<size_t>(r) * L; }
};

// Random initial state ("hot start"), again keyed by global site so every
// backend starts from the same configuration.
inline void init_random(Lattice& lat, std::uint64_t seed) {
    for (int r = 1; r <= lat.nrows; ++r) {
        const std::uint64_t g = static_cast<std::uint64_t>(lat.row0 + r - 1);
        for (int c = 0; c < lat.L; ++c) {
            const double u = site_uniform(g * lat.L + c, ~0ull, 0, seed);
            lat.row(r)[c] = u < 0.5 ? 1 : -1;
        }
    }
}

// Acceptance probabilities exp(-beta * dE) for dE = 4 and 8 (J = 1, k_B = 1).
struct Acceptance {
    double p4, p8;
    explicit Acceptance(double beta) : p4(std::exp(-4.0 * beta)), p8(std::exp(-8.0 * beta)) {}
};

// Update every owned site of colour `parity`. Ghost rows must be current.
// Returns nothing; the RNG draw is skipped when dE <= 0 (always accepted),
// which keeps results identical across backends because the counter, not a
// stream position, determines each draw.
inline void half_sweep(Lattice& lat, int parity, std::uint64_t sweep, std::uint64_t seed, const Acceptance& acc) {
    const int L = lat.L;
#pragma omp parallel for schedule(static)
    for (int r = 1; r <= lat.nrows; ++r) {
        const std::int64_t g = lat.row0 + r - 1;
        std::int8_t* cur = lat.row(r);
        const std::int8_t* up = lat.row(r - 1);
        const std::int8_t* dn = lat.row(r + 1);
        for (int c = static_cast<int>((g + parity) & 1); c < L; c += 2) {
            const int left = c == 0 ? L - 1 : c - 1;
            const int right = c == L - 1 ? 0 : c + 1;
            const int nn = up[c] + dn[c] + cur[left] + cur[right];
            const int dE = 2 * cur[c] * nn;  // in {-8, -4, 0, 4, 8}
            if (dE <= 0) {
                cur[c] = static_cast<std::int8_t>(-cur[c]);
            } else {
                const double u = site_uniform(static_cast<std::uint64_t>(g) * L + c, sweep, parity, seed);
                if (u < (dE == 4 ? acc.p4 : acc.p8)) cur[c] = static_cast<std::int8_t>(-cur[c]);
            }
        }
    }
}

// Optimised kernel. Profiling the baseline showed the Philox call (10 rounds
// of 32x32 multiplies per update) dominating the cost. Here one Philox call
// yields four 32-bit outputs that serve four consecutive same-colour sites,
// and acceptance is an integer compare against precomputed 2^32 * exp(-beta dE)
// thresholds. The RNG block for a site is still a pure function of its global
// position, so backends remain bit-identical to each other (the stream differs
// from the baseline kernel, so the two kernels give different but equally
// valid trajectories).
struct Thresholds {
    std::uint64_t t4, t8;
    explicit Thresholds(const Acceptance& a)
        : t4(static_cast<std::uint64_t>(a.p4 * 4294967296.0)), t8(static_cast<std::uint64_t>(a.p8 * 4294967296.0)) {}
};

inline void half_sweep_blocked(Lattice& lat, int parity, std::uint64_t sweep, std::uint64_t seed,
                               const Thresholds& thr) {
    const int L = lat.L, half = L / 2;
    const Philox4x32Key key = {static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32)};
    const std::uint32_t c2 = static_cast<std::uint32_t>(sweep);
    const std::uint32_t c3 = static_cast<std::uint32_t>((sweep >> 32) << 1) | static_cast<std::uint32_t>(parity) |
                             0x80000000u;  // high bit separates this stream from the baseline kernel
#pragma omp parallel for schedule(static)
    for (int r = 1; r <= lat.nrows; ++r) {
        const std::int64_t g = lat.row0 + r - 1;
        std::int8_t* cur = lat.row(r);
        const std::int8_t* up = lat.row(r - 1);
        const std::int8_t* dn = lat.row(r + 1);
        const int start = static_cast<int>((g + parity) & 1);
        std::uint64_t q = static_cast<std::uint64_t>(g) * half;  // global index among this colour's sites
        Philox4x32Ctr rnd{};
        for (int j = 0; j < half; ++j, ++q) {
            if (j == 0 || (q & 3) == 0) {
                const std::uint64_t block = q >> 2;
                rnd = philox4x32_10({static_cast<std::uint32_t>(block), static_cast<std::uint32_t>(block >> 32), c2, c3},
                                    key);
            }
            const int c = start + 2 * j;
            const int left = c == 0 ? L - 1 : c - 1;
            const int right = c == L - 1 ? 0 : c + 1;
            const int dE = 2 * cur[c] * (up[c] + dn[c] + cur[left] + cur[right]);
            const std::uint64_t t = dE <= 0 ? 4294967296ull : (dE == 4 ? thr.t4 : thr.t8);
            if (rnd[q & 3] < t) cur[c] = static_cast<std::int8_t>(-cur[c]);
        }
    }
}

// Same random numbers as half_sweep_blocked, generated a whole row at a time.
// Phase 1 runs Philox over all blocks of the row in structure-of-arrays form,
// a dependency-free loop the compiler vectorises (AVX2/AVX-512 have the
// 32x32->64 multiply Philox needs). Phase 2 is the spin update. Bit-identical
// to half_sweep_blocked, which the backend test checks.
inline void philox_rows_soa(std::uint64_t first_block, int nblk, std::uint32_t c2, std::uint32_t c3,
                            Philox4x32Key key0, std::uint32_t* out) {
    constexpr std::uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    constexpr std::uint32_t W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
#pragma omp simd
    for (int b = 0; b < nblk; ++b) {
        const std::uint64_t block = first_block + static_cast<std::uint64_t>(b);
        std::uint32_t x0 = static_cast<std::uint32_t>(block), x1 = static_cast<std::uint32_t>(block >> 32), x2 = c2,
                      x3 = c3;
        std::uint32_t k0 = key0[0], k1 = key0[1];
        for (int round = 0; round < 10; ++round) {
            const std::uint64_t p0 = static_cast<std::uint64_t>(M0) * x0;
            const std::uint64_t p1 = static_cast<std::uint64_t>(M1) * x2;
            const std::uint32_t hi0 = static_cast<std::uint32_t>(p0 >> 32), lo0 = static_cast<std::uint32_t>(p0);
            const std::uint32_t hi1 = static_cast<std::uint32_t>(p1 >> 32), lo1 = static_cast<std::uint32_t>(p1);
            x0 = hi1 ^ x1 ^ k0;
            x1 = lo1;
            x2 = hi0 ^ x3 ^ k1;
            x3 = lo0;
            k0 += W0;
            k1 += W1;
        }
        out[4 * b + 0] = x0;
        out[4 * b + 1] = x1;
        out[4 * b + 2] = x2;
        out[4 * b + 3] = x3;
    }
}

inline void half_sweep_batched(Lattice& lat, int parity, std::uint64_t sweep, std::uint64_t seed,
                               const Thresholds& thr) {
    const int L = lat.L, half = L / 2;
    const Philox4x32Key key = {static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32)};
    const std::uint32_t c2 = static_cast<std::uint32_t>(sweep);
    const std::uint32_t c3 = static_cast<std::uint32_t>((sweep >> 32) << 1) | static_cast<std::uint32_t>(parity) |
                             0x80000000u;
    const std::uint64_t tab[3] = {4294967296ull, thr.t4, thr.t8};  // indexed by max(dE, 0) / 4
#pragma omp parallel
    {
        std::vector<std::uint32_t> rnd(static_cast<size_t>(half / 4 + 2) * 4);
#pragma omp for schedule(static)
        for (int r = 1; r <= lat.nrows; ++r) {
            const std::int64_t g = lat.row0 + r - 1;
            std::int8_t* cur = lat.row(r);
            const std::int8_t* up = lat.row(r - 1);
            const std::int8_t* dn = lat.row(r + 1);
            const int start = static_cast<int>((g + parity) & 1);
            const std::uint64_t q0 = static_cast<std::uint64_t>(g) * half;
            const std::uint64_t b0 = q0 >> 2, b1 = (q0 + half - 1) >> 2;
            philox_rows_soa(b0, static_cast<int>(b1 - b0 + 1), c2, c3, key, rnd.data());
            const std::uint32_t* rr = rnd.data() + (q0 & 3);  // rr[j] is the draw for colour-site j of this row
            // Branchless update: near T_c acceptance is close to a coin flip, so
            // an if-statement mispredicts constantly. idx = max(dE, 0) / 4 and
            // the flip is a multiply by +-1, which compiles to straight-line code.
            auto update_site = [&](int j, int c, int left, int right) {
                const int dE = 2 * cur[c] * (up[c] + dn[c] + cur[left] + cur[right]);
                const int idx = (dE + (dE < 0 ? -dE : dE)) >> 3;
                const int flip = rr[j] < tab[idx];
                cur[c] = static_cast<std::int8_t>(cur[c] * (1 - 2 * flip));
            };
            // Wrap-around neighbours only occur at the two ends of the row.
            update_site(0, start, start == 0 ? L - 1 : start - 1, start + 1);
            for (int j = 1; j < half - 1; ++j) {
                const int c = start + 2 * j;
                update_site(j, c, c - 1, c + 1);
            }
            const int cl = start + 2 * (half - 1);
            update_site(half - 1, cl, cl - 1, cl == L - 1 ? 0 : cl + 1);
        }
    }
}

// Periodic ghost rows for a single owner of the whole lattice.
inline void fill_periodic_ghosts(Lattice& lat) {
    std::copy(lat.row(lat.nrows), lat.row(lat.nrows) + lat.L, lat.row(0));
    std::copy(lat.row(1), lat.row(1) + lat.L, lat.row(lat.nrows + 1));
}

// Sums over owned sites: magnetisation and bond energy (each bond counted once
// via the right and down neighbours). Requires a current lower ghost row.
struct Sums {
    std::int64_t M = 0;
    std::int64_t E = 0;
};

inline Sums local_sums(const Lattice& lat) {
    Sums out;
    const int L = lat.L;
    std::int64_t M = 0, E = 0;
#pragma omp parallel for reduction(+ : M, E) schedule(static)
    for (int r = 1; r <= lat.nrows; ++r) {
        const std::int8_t* cur = lat.row(r);
        const std::int8_t* dn = lat.row(r + 1);
        for (int c = 0; c < L; ++c) {
            const int right = c == L - 1 ? 0 : c + 1;
            M += cur[c];
            E -= cur[c] * (cur[right] + dn[c]);
        }
    }
    out.M = M;
    out.E = E;
    return out;
}

// FNV-1a hash of the owned rows, used to check backends agree bit for bit.
inline std::uint64_t fnv1a(const std::int8_t* p, size_t n, std::uint64_t h = 1469598103934665603ull) {
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<std::uint8_t>(p[i]);
        h *= 1099511628211ull;
    }
    return h;
}

}  // namespace ising
