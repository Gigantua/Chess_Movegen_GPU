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

    __shared__ uint64_t shr_HO[64];
    __shared__ uint64_t shr_VE[64];
    __shared__ uint64_t shr_D1[64];
    __shared__ uint64_t shr_D2[64];
    __shared__ uint64_t shr_LO[64];

    __inline__ __device__ void Prepare(unsigned int threadIdx)
    {
        if (threadIdx < 64)
        {
            const uint64_t s_bit = 1ull << threadIdx;

            shr_HO[threadIdx] = dir_HO(threadIdx) ^ s_bit;
            shr_VE[threadIdx] = dir_VE(threadIdx) ^ s_bit;
            shr_D1[threadIdx] = dir_D1(threadIdx) ^ s_bit;
            shr_D2[threadIdx] = dir_D2(threadIdx) ^ s_bit;
            shr_LO[threadIdx] = s_bit - 1;
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
        const uint64_t lower = occ &  shr_LO[sq];
        const uint64_t upper = occ & ~shr_LO[sq];
        const uint64_t ho = shr_HO[sq];
        const uint64_t ve = shr_VE[sq];

        return line_attack(ho & lower, ho & upper, ho) |
               line_attack(ve & lower, ve & upper, ve);
    }

    BitFunction Rook(int sq, uint64_t occ)
    {
        const uint64_t lower = occ &  shr_LO[sq];
        const uint64_t upper = occ & ~shr_LO[sq];
        const uint64_t ho = shr_D1[sq];
        const uint64_t ve = shr_D2[sq];

        return line_attack(ho & lower, ho & upper, ho) |
               line_attack(ve & lower, ve & upper, ve);
    }

    BitFunction Queen(int sq, uint64_t occ)
    {
        return Bishop(sq, occ) | Rook(sq, occ);
    }

#undef BitFunction
#undef GetLower
#undef GetUpper
#undef dir_HO
#undef dir_VE
#undef dir_D1
#undef dir_D2

}