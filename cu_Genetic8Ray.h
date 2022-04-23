#pragma once
/**
 * Genetic8Ray.hpp
 *
 * Copyright © 2022 Daniel Inführ
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * For Questions and allowance to use this code contact under daniel.infuehr(@)live.de
 *
 * Maintained by Daniel Inführ, 2022
 *
 * @file Genetic8Ray.hpp
 * @author Daniel Inführ
 * @copyright 2022
 * @section License
 */

 //This was created by writing a custom C++ Code generator that 
 //tries every abstract syntax tree that is legal with a list of 16 c++ code tokens. 


#include <stdint.h>
#include <array>
#include <type_traits>
#include "cu_Common.h"

#define BitFunction __inline__ __device__ uint64_t
#define dir_HO(X) (0xFFull << (X & 56))
#define dir_VE(X) (0x0101010101010101ull << (X & 7))
#define dir_D1(X) (mask_shift<0x8040201008040201ull>((X & 7) - (X >> 3)))
#define dir_D2(X) (mask_shift<0x0102040810204080ull>(7 - (X & 7) - (X >> 3)))
#define GetLower(X) ((1ull << X) - 1)
#define GetUpper(X) (0xFFFFFFFFFFFFFFFE << (X))
#define bit_reverse(X) __brevll(X)

namespace Genetic8Ray
{
	template<uint64_t bb>
	constexpr BitFunction mask_shift(int ranks) {
		return ranks > 0 ? bb >> (ranks << 3) : bb << -(ranks << 3);
	}

	BitFunction SolveLineUpper_HO(uint64_t occ, uint64_t mask)
	{
		occ &= mask;
		return (occ ^ (occ - 1ull)) & mask;
	}

	BitFunction SolveLineLower_HO(uint64_t occ, uint64_t mask)
	{
		occ &= mask;
		return (occ ^ bit_reverse((bit_reverse(occ) - 1ull))) & mask;
	}

	BitFunction SolveLineUpper(uint64_t occ, uint64_t mask)
	{
		occ &= mask;
		return ((occ - 1ull) << 1ull) & mask;
	}

	BitFunction SolveLineLower(uint64_t occ, uint64_t mask)
	{
		occ &= mask;
		return (bit_reverse((bit_reverse(occ) - 1ull)) >> 1ull) & mask;
	}
	__shared__ uint64_t shr_HO[64];
	__shared__ uint64_t shr_VE[64];
	__shared__ uint64_t shr_D1[64];
	__shared__ uint64_t shr_D2[64];
	__shared__ uint64_t shr_lower[64];
	__shared__ uint64_t shr_upper[64];

	BitFunction Rook(int sq, uint64_t occ)
	{
		const uint64_t lower = shr_lower[sq];
		const uint64_t upper = shr_upper[sq];
		const uint64_t ho = shr_HO[sq];
		const uint64_t ve = shr_VE[sq];

		return SolveLineUpper_HO(occ, upper & ho) | SolveLineLower_HO(occ, lower & ho) |
			   SolveLineUpper(occ, upper & ve) | SolveLineLower(occ, lower & ve);
	}

	BitFunction Bishop(int sq, uint64_t occ)
	{
		const uint64_t lower = shr_lower[sq];
		const uint64_t upper = shr_upper[sq];
		const uint64_t d1 = shr_D1[sq];
		const uint64_t d2 = shr_D2[sq];

		return SolveLineUpper(occ, upper & d1) | SolveLineLower(occ, lower & d1) |
			   SolveLineUpper(occ, upper & d2) | SolveLineLower(occ, lower & d2);
	}

	BitFunction Queen(int sq, uint64_t occ)
	{
		return Bishop(sq, occ) | Rook(sq, occ);
	}


	__inline__ __device__ void Prepare(unsigned int threadIdx)
	{
		if (threadIdx < 64)
		{
			const uint64_t s_bit = 1ull << threadIdx;
		
			shr_HO[threadIdx] = dir_HO(threadIdx) ^ s_bit;
			shr_VE[threadIdx] = dir_VE(threadIdx) ^ s_bit;
			shr_D1[threadIdx] = dir_D1(threadIdx) ^ s_bit;
			shr_D2[threadIdx] = dir_D2(threadIdx) ^ s_bit;
			shr_lower[threadIdx] = GetLower(threadIdx);
			shr_upper[threadIdx] = GetUpper(threadIdx);
		}
		__syncthreads();
	}
}

#undef BitFunction 
#undef dir_HO 
#undef dir_VE 
#undef dir_D1 
#undef dir_D2 
#undef GetLower 
#undef GetUpper 
#undef bit_reverse 