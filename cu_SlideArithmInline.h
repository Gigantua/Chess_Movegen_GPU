#pragma once
#include <array>
#include <cstdint>

#include "cu_Common.h"

namespace SlideArithmInline
{
	template<uint64_t bb>
	static constexpr __inline__ __device__ uint64_t mask_shift(int ranks) {
		return ranks > 0 ? bb >> (ranks << 3) : bb << -(ranks << 3);
	}
	
    template<int dir>
    static constexpr __inline__ __device__ uint64_t dirMask(int square) {

        if constexpr (dir == 0) return (0xFFull << (square & 56)) ^ (1ull << square); //HORIZONTAL
        else if constexpr (dir == 1) return (0x0101010101010101ull << (square & 7)) ^ (1ull << square); //VERTICAL
        else 
        {
            int file = square & 7; int rank = square >> 3;
            if constexpr (dir == 2) return (mask_shift<0x8040201008040201ull>(file - rank)) ^ (1ull << square); //Diagonal
            else return (mask_shift<0x0102040810204080ull>(7 - file - rank)) ^ (1ull << square); //Antidiagonal
        }
		return 0;
    }

	constexpr auto Size = 0;

	__inline__ __device__ uint64_t bzhi(uint64_t src, int idx) {
		return src & (1 << idx) - 1;
	}

	__inline__ __device__ uint64_t blsmsk(uint64_t x) {
		return x ^ (x - 1);
	}

	__inline__ __device__ uint64_t countl_zero(uint64_t x) {
		return __clzll(x);
	}


	/* Start of code */
	__inline__ __device__
	static uint64_t slide_arithmetic(uint32_t p, uint64_t block) {
		//BZHI
		//[src & (1 << inx) - 1] ;
		// split the line into upper and lower rays
		uint64_t mask = bzhi(block, p);

		// for the bottom we use CLZ + a shift to fill in from the top
		uint64_t blocked_down = 0x7FFFFFFFFFFFFFFFull >> countl_zero(block & mask | 1ull);

		//_blsmsk_u64 = X^X-1
		// the intersection of the two is the move set after masking with the line
		return (blsmsk(block & ~mask) ^ blocked_down);
	}

	__inline__ __device__
	static uint64_t Queen(uint32_t s, uint64_t occ)
	{
		const uint64_t p = 1ull << s;
		const uint64_t h =  dirMask<0>(s) ^ p;
		const uint64_t v =  dirMask<1>(s) ^ p;
		const uint64_t d1 = dirMask<2>(s) ^ p;
		const uint64_t d2 = dirMask<3>(s) ^ p;
		return slide_arithmetic(s, h & occ) & h
			^ slide_arithmetic(s, v & occ) & v
			^ slide_arithmetic(s, d1 & occ) & d1
			^ slide_arithmetic(s, d2 & occ) & d2;
	}
}