#pragma once
/* M42 - A small library for Bitboard attack mask generation, by Syed Fahad
 *
 * Usage: Just include this file as a normal header, and m42.cpp into your
 * source code, and then call M42::init() at your program startup.
 * THIS IS IMPORTANT, you've to call M42::init() in the main() function of
 * your program, to use this library.
 * Read the documentation to know what every function does.
 *
 * Developer contact: sydfhd at gmail dot com
 *
 * License: MIT License (see https://mit-license.org/)
 */

 // Datum  : 29.01.2022
 // Author : Constexpr Inlined SBAMG Version: Daniel Infuehr

#pragma once
#include <stdint.h>
#include <array>
#include "cu_Common.h"

#	define BitFunction __inline__ __device__ uint64_t
#	define dir_HO(X) (0xFFull << (X & 56))
#	define dir_VE(X) (0x0101010101010101ull << (X & 7))
#	define dir_D1(X) (mask_shift<0x8040201008040201ull>(    (X & 7) - (X >> 3)))
#	define dir_D2(X) (mask_shift<0x0102040810204080ull>(7 - (X & 7) - (X >> 3)))
namespace SBAMG
{
    template<uint64_t bb>
    constexpr BitFunction mask_shift(int ranks) {
        return ranks > 0 ? bb >> (ranks << 3) : bb << -(ranks << 3);
    }

    static constexpr __device__ uint64_t outersquare(int square) {
        return (0x81ull << (square & 56)) & ~(1ull << square) | (square == 0); //Probably can be optimised - 2ull << (square >> 1) === (1ull << (square)) - (square == 0)
    }

    static constexpr __device__ uint64_t outersquare_file(int square) {
        return (0x0100000000000001ULL << (square & 7)) & ~(1ull << square); //VERTICAL
    }

    static constexpr __device__ uint64_t ThisAndNextSq(int sq)
    {
        return 3ULL << sq;
    };
    static constexpr __device__ uint64_t PrevSquares(int sq)
    {
        return ((1ULL << sq) - 1) | (sq == 0);
    };


    static __device__ int msb(uint64_t value)
    {
        return 63 - __clzll(value);
    }


    static constexpr __device__ uint64_t rank_attacks(int sq, uint64_t occ)
    {
        const uint64_t rankmask = dir_HO(sq);
        occ = (occ & rankmask) | outersquare(sq);
        return ((occ - ThisAndNextSq(msb(occ & PrevSquares(sq)))) ^ occ)
            & rankmask;
    }

    static constexpr __device__ uint64_t file_attacks(int sq, uint64_t occ)
    {
        const uint64_t filemask = dir_VE(sq);
        occ = (occ & filemask) | outersquare_file(sq);
        return ((occ - ThisAndNextSq(msb(occ & PrevSquares(sq)))) ^ occ)
            & filemask;
    }

    static constexpr __device__ uint64_t byteswap_constexpr(uint64_t b) {
        //Todo: Test for __brevll 
        b = ((b >> 8) & 0x00FF00FF00FF00FFULL) | ((b & 0x00FF00FF00FF00FFULL) << 8);
        b = ((b >> 16) & 0x0000FFFF0000FFFFULL) | ((b & 0x0000FFFF0000FFFFULL) << 16);
        return (b >> 32) | (b << 32);
    }

    static constexpr __device__ uint64_t byteswap(uint64_t b) {
        return byteswap_constexpr(b);
    }

    // NORMAL VERSION
    static constexpr __device__ uint64_t diag_attacks(int sq, uint64_t occ)
    {
        const auto diagmask = dir_D1(sq);
        occ &= diagmask;
        return ((occ - (1ull << sq)) ^ byteswap(byteswap(occ) - (1ull << (sq ^ 56))))
            & diagmask;
    }

    static constexpr __device__ uint64_t adiag_attacks(int sq, uint64_t occ)
    {
        const auto adiagmask = dir_D2(sq);
        occ &= adiagmask;
        return ((occ - (1ull << sq)) ^ byteswap(byteswap(occ) - (1ull << (sq ^ 56))))
            & adiagmask;
    }

    static constexpr __device__ uint64_t Bishop(int sq, uint64_t occ)
    {
        return diag_attacks(sq, occ) | adiag_attacks(sq, occ);
    }

    static constexpr __device__ uint64_t Rook(int sq, uint64_t occ)
    {
        return file_attacks(sq, occ) | rank_attacks(sq, occ);
    }

    static constexpr __device__ uint64_t Queen(int sq, uint64_t occ)
    {
        return Bishop(sq, occ) | Rook(sq, occ);
    }
}
#undef BitFunction
#undef dir_HO
#undef dir_VE
#undef dir_D1
#undef dir_D2