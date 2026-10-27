// Wolff single-cluster algorithm (Wolff, PRL 62, 361, 1989).
//
// Grows a cluster of aligned spins from a random seed site, adding each aligned
// neighbour with probability p_add = 1 - exp(-2 beta), then flips the whole
// cluster. Near T_c this decorrelates configurations far faster than local
// Metropolis updates, which is the comparison in the autocorrelation study.
#pragma once
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace ising {

class Wolff {
public:
    Wolff(int L, double T, std::uint64_t seed)
        : L_(L), N_(L * L), p_add_(1.0 - std::exp(-2.0 / T)), s_(N_), rng_(seed), stack_() {
        stack_.reserve(N_);
        std::uniform_int_distribution<int> coin(0, 1);
        for (auto& x : s_) x = coin(rng_) ? 1 : -1;
    }

    // Build and flip one cluster; returns its size.
    int step() {
        std::uniform_int_distribution<int> pick(0, N_ - 1);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        const int seed = pick(rng_);
        const std::int8_t spin = s_[seed];
        s_[seed] = static_cast<std::int8_t>(-spin);  // flip on insertion marks it visited
        stack_.clear();
        stack_.push_back(seed);
        int size = 1;
        while (!stack_.empty()) {
            const int i = stack_.back();
            stack_.pop_back();
            const int r = i / L_, c = i % L_;
            const int nbr[4] = {((r + L_ - 1) % L_) * L_ + c, ((r + 1) % L_) * L_ + c, r * L_ + (c + L_ - 1) % L_,
                                r * L_ + (c + 1) % L_};
            for (int j : nbr) {
                if (s_[j] == spin && u(rng_) < p_add_) {
                    s_[j] = static_cast<std::int8_t>(-spin);
                    stack_.push_back(j);
                    ++size;
                }
            }
        }
        return size;
    }

    // Per-site energy and magnetisation.
    double energy() const {
        std::int64_t E = 0;
        for (int r = 0; r < L_; ++r)
            for (int c = 0; c < L_; ++c) {
                const int i = r * L_ + c;
                E -= s_[i] * (s_[r * L_ + (c + 1) % L_] + s_[((r + 1) % L_) * L_ + c]);
            }
        return static_cast<double>(E) / N_;
    }
    double magnetization() const {
        std::int64_t M = 0;
        for (auto x : s_) M += x;
        return static_cast<double>(M) / N_;
    }
    int sites() const { return N_; }

private:
    int L_, N_;
    double p_add_;
    std::vector<std::int8_t> s_;
    std::mt19937_64 rng_;
    std::vector<int> stack_;
};

}  // namespace ising
