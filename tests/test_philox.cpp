// Known-answer tests for Philox4x32-10 (vectors from the Random123 distribution, kat_vectors).
#include <cstdio>

#include "philox.hpp"

int main() {
    using namespace ising;
    struct Case {
        Philox4x32Ctr ctr;
        Philox4x32Key key;
        Philox4x32Ctr expect;
    };
    const Case cases[] = {
        {{0u, 0u, 0u, 0u}, {0u, 0u}, {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}},
        {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu},
         {0xffffffffu, 0xffffffffu},
         {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}},
        {{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u},
         {0xa4093822u, 0x299f31d0u},
         {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}},
    };
    int failures = 0;
    for (const auto& c : cases) {
        const auto got = philox4x32_10(c.ctr, c.key);
        if (got != c.expect) {
            ++failures;
            std::printf("FAIL: got %08x %08x %08x %08x\n", got[0], got[1], got[2], got[3]);
        }
    }
    std::printf("philox known-answer tests: %s\n", failures ? "FAILED" : "passed");
    return failures ? 1 : 0;
}
