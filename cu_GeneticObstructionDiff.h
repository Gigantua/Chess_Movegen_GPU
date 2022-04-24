#pragma once
/**
 * cu_GeneticObstructionDiff.hpp
 *
 * Copyright © 2022 Daniel Inführ
 *
 * This file is under MIT licence and may be used in any piece of code if you keep this unedited header in the sourcefile
 * Created by C++ Abstract Sytax Tree Sifting - improvement of ObstructionDifference
 * For Questions and contact under daniel.infuehr(@)live.de
 *
 * Maintained by Daniel Inführ, 2022
 *
 * @file cu_GeneticObstructionDiff.hpp
 * @author Daniel Inführ
 * @copyright 16.01.2022
 * @section License
 */

#include <stdint.h>
#include <array>
#include "cu_Common.h"

namespace GeneticObstructionDifference {

#define BitFunction __inline__ __device__ uint64_t
#define GetLower(S) ((1ull << S) - 1)
#define GetUpper(S) (0xFFFFFFFFFFFFFFFF << (S))
#define dir_HO(X) (0xFFull << (X & 56))
#define dir_VE(X) (0x0101010101010101ull << (X & 7))
#define dir_D1(X) (mask_shift<0x8040201008040201ull>((X & 7) - (X >> 3)))
#define dir_D2(X) (mask_shift<0x0102040810204080ull>(7 - (X & 7) - (X >> 3)))

    template<uint64_t bb>
    static constexpr BitFunction mask_shift(int ranks) {
        return ranks > 0 ? bb >> (ranks << 3) : bb << -(ranks << 3);
    }

    struct RayMask 
    {
        uint64_t HO, VE, D1, D2, LO, UP;
    };

    __shared__ RayMask shr[64];
    __inline__ __device__ void Prepare(unsigned int threadIdx)
    {
        if (threadIdx < 64)
        {
            shr[threadIdx] = {
                dir_HO(threadIdx),
                dir_VE(threadIdx),
                dir_D1(threadIdx),
                dir_D2(threadIdx),
                (1ull << threadIdx) - 1,
                (0xFFFFFFFFFFFFFFFE << threadIdx)
            };
        }
        __syncthreads();
    }

    BitFunction line_attack(uint64_t lower, uint64_t upper, uint64_t mask)
    {
        const uint64_t msb = 0x8000000000000000ull >> __clzll(lower | 1);
        return (mask & (upper ^ (upper - msb)));
    }

    BitFunction Bishop(int sq, uint64_t occ)
    {
        const auto& r = shr[sq];
        return line_attack(r.HO & occ & r.LO, r.HO & occ & r.UP, r.HO) |
               line_attack(r.VE & occ & r.LO, r.VE & occ & r.UP, r.VE);
    }

    BitFunction Rook(int sq, uint64_t occ)
    {
        const auto& r = shr[sq];
        return line_attack(r.D1 & occ & r.LO, r.D1 & occ & r.UP, r.D1) |
               line_attack(r.D2 & occ & r.LO, r.D2 & occ & r.UP, r.D2);
    }

    BitFunction Queen(int sq, uint64_t occ)
    {
        const auto& r = shr[sq];
        return line_attack(r.HO & occ & r.LO, r.HO & occ & r.UP, r.HO) |
               line_attack(r.VE & occ & r.LO, r.VE & occ & r.UP, r.VE) |
               line_attack(r.D1 & occ & r.LO, r.D1 & occ & r.UP, r.D1) |
               line_attack(r.D1 & occ & r.LO, r.D1 & occ & r.UP, r.D2);
    }

#undef BitFunction
#undef GetLower
#undef GetUpper
#undef dir_HO
#undef dir_VE
#undef dir_D1
#undef dir_D2

}