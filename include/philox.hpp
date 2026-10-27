// Philox4x32-10 counter-based random number generator (Salmon et al., SC'11).
//
// A counter-based generator maps (counter, key) -> random bits with no hidden
// state. Every spin update draws its random number from a counter built out of
// (global site index, sweep, sublattice), so the stream a site sees does not
// depend on how the lattice is split across threads or MPI ranks. That is what
// makes the serial, OpenMP and MPI backends produce bit-identical results.
#pragma once
#include <array>
#include <cstdint>

namespace ising {

using Philox4x32Ctr = std::array<std::uint32_t, 4>;
using Philox4x32Key = std::array<std::uint32_t, 2>;

inline void mulhilo32(std::uint32_t a, std::uint32_t b, std::uint32_t& hi, std::uint32_t& lo) {
    const std::uint64_t p = static_cast<std::uint64_t>(a) * b;
    hi = static_cast<std::uint32_t>(p >> 32);
    lo = static_cast<std::uint32_t>(p);
}

inline Philox4x32Ctr philox4x32_10(Philox4x32Ctr c, Philox4x32Key k) {
    constexpr std::uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    constexpr std::uint32_t W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
    for (int round = 0; round < 10; ++round) {
        std::uint32_t hi0, lo0, hi1, lo1;
        mulhilo32(M0, c[0], hi0, lo0);
        mulhilo32(M1, c[2], hi1, lo1);
        c = {hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0};
        k[0] += W0;
        k[1] += W1;
    }
    return c;
}

// Uniform double in [0, 1) for one spin update.
inline double site_uniform(std::uint64_t site, std::uint64_t sweep, int parity, std::uint64_t seed) {
    const Philox4x32Ctr ctr = {static_cast<std::uint32_t>(site), static_cast<std::uint32_t>(site >> 32),
                               static_cast<std::uint32_t>(sweep),
                               static_cast<std::uint32_t>((sweep >> 32) << 1) | static_cast<std::uint32_t>(parity)};
    const Philox4x32Key key = {static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32)};
    const Philox4x32Ctr r = philox4x32_10(ctr, key);
    return r[0] * 2.3283064365386963e-10;  // 2^-32
}

}  // namespace ising
