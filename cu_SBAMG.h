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

namespace SBAMG
{
    template<uint64_t bb>
    static constexpr __device__ uint64_t mask_shift(int ranks) {
        return ranks > 0 ? bb >> (ranks << 3) : bb << -(ranks << 3);
    }
    template<int dir>
    static constexpr __device__ uint64_t dirMask(int square) {

        if constexpr (dir == 0) return (0xFFull << (square & 56)) ^ (1ull << square); //HORIZONTAL
        else if constexpr (dir == 1) return (0x0101010101010101ull << (square & 7)) ^ (1ull << square); //VERTICAL
        else 
        {
            int file = square & 7;
            int rank = square >> 3;
            if constexpr (dir == 2) return (mask_shift<0x8040201008040201ull>(file - rank)) ^ (1ull << square); //Diagonal
            else return (mask_shift<0x0102040810204080ull>(7 - file - rank)) ^ (1ull << square); //Antidiagonal
        }
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


    static __device__ uint64_t msb(uint64_t value)
    {
        return 63ull - __clzll(value);
    }


    static constexpr __device__ uint64_t rank_attacks(int sq, uint64_t occ)
    {
        const uint64_t rankmask = dirMask<0>(sq);
        occ = (occ & rankmask) | outersquare(sq);
        return ((occ - ThisAndNextSq(msb(occ & PrevSquares(sq)))) ^ occ)
            & rankmask;
    }

    static constexpr __device__ uint64_t file_attacks(int sq, uint64_t occ)
    {
        const uint64_t filemask = dirMask<1>(sq);
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
        const auto diagmask = dirMask<2>(sq);
        occ &= diagmask;
        return ((occ - (1ull << sq)) ^ byteswap(byteswap(occ) - (1ull << (sq ^ 56))))
            & diagmask;
    }

    static constexpr __device__ uint64_t adiag_attacks(int sq, uint64_t occ)
    {
        const auto adiagmask = dirMask<3>(sq);
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