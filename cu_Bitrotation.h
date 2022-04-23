#pragma once

//(c) Gerd Isenberg, Aleks Peshkov 2007
//Subtracting a Rook from a Blocking Piece - https://www.chessprogramming.org/Subtracting_a_Rook_from_a_Blocking_Piece

#include <stdint.h>
#include <array>
#include <type_traits>
#include "cu_Common.h"

//Improvement by Daniel Inführ 2022
//Update 2022: This is the fastest algorithm known on any platform by far! http://www.talkchess.com/forum3/viewtopic.php?f=7&t=79078&start=20
//bitswap intrinsics harmonizes all 4 possible paths

namespace Bitrotation {
#	define BitFunction __inline__ __device__ uint64_t
#	define dir_HO(X) (0xFFull << (X & 56))
#	define dir_VE(X) (0x0101010101010101ull << (X & 7))
#	define dir_D1(X) (mask_shift<0x8040201008040201ull>(    (X & 7) - (X >> 3)))
#	define dir_D2(X) (mask_shift<0x0102040810204080ull>(7 - (X & 7) - (X >> 3)))
#	define bitswap(X) __brevll(X)

	template<uint64_t bb>
	constexpr BitFunction mask_shift(int ranks) {
		return ranks > 0 ? bb >> (ranks << 3) : bb << -(ranks << 3);
	}

	/* Generate attack using the hyperbola quintessence approach */
	/* ((occ - (1ull << sq)) ^ bit_reverse((bit_reverse(occ) - bit_reverse((1ull << sq))))) */
	BitFunction attack(uint64_t occ, uint64_t mask, uint64_t s_bit, uint64_t s_rev)
	{
		const uint64_t o = (occ & mask);
		return ((o - s_bit) ^ bitswap(bitswap(o) - s_rev)) & mask;
	}

	__shared__ uint64_t shr_HO[256];
	__shared__ uint64_t shr_VE[256];
	__shared__ uint64_t shr_D1[256];
	__shared__ uint64_t shr_D2[256];

	BitFunction Queen(int s, const uint64_t occ) {
		const uint64_t s_bit = 1ull << s;
		const uint64_t s_rev = (1ull << (s ^ 56));
		
		return (attack(occ, shr_HO[s], s_bit, s_rev))
			 | (attack(occ, shr_VE[s], s_bit, s_rev))
			 | (attack(occ, shr_D1[s], s_bit, s_rev))
			 | (attack(occ, shr_D2[s], s_bit, s_rev));
	}

	__inline__ __device__ void Prepare(unsigned int threadIdx)
	{
		int sq = threadIdx % 64;
		shr_HO[threadIdx] = dir_HO(sq) ^ (1ull << sq);
		shr_VE[threadIdx] = dir_VE(sq) ^ (1ull << sq);
		shr_D1[threadIdx] = dir_D1(sq) ^ (1ull << sq);
		shr_D2[threadIdx] = dir_D2(sq) ^ (1ull << sq);

		__syncthreads();
	}

#undef BitFunction
#undef dir_HO
#undef dir_VE
#undef dir_D1
#undef dir_D2
#undef bitswap
}


/*
* Cuda is not the same as x64. Uncommenting this code is 3x slower than direct mask_shift calculation!
__constant__ static const uint64_t d1[] = { 9241421688590303745ull, 36099303471055874ull, 141012904183812ull, 550831656968ull, 2151686160ull, 8405024ull, 32832ull, 128ull, 4620710844295151872ull, 9241421688590303745ull, 36099303471055874ull, 141012904183812ull, 550831656968ull, 2151686160ull, 8405024ull, 32832ull, 2310355422147575808ull,
		4620710844295151872ull, 9241421688590303745ull, 36099303471055874ull, 141012904183812ull, 550831656968ull, 2151686160ull, 8405024ull, 1155177711073755136ull, 2310355422147575808ull, 4620710844295151872ull, 9241421688590303745ull, 36099303471055874ull, 141012904183812ull, 550831656968ull, 2151686160ull, 577588855528488960ull,
		1155177711073755136ull, 2310355422147575808ull, 4620710844295151872ull, 9241421688590303745ull, 36099303471055874ull, 141012904183812ull, 550831656968ull, 288794425616760832ull, 577588855528488960ull, 1155177711073755136ull, 2310355422147575808ull, 4620710844295151872ull, 9241421688590303745ull, 36099303471055874ull, 141012904183812ull,
		144396663052566528ull, 288794425616760832ull, 577588855528488960ull, 1155177711073755136ull, 2310355422147575808ull, 4620710844295151872ull, 9241421688590303745ull, 36099303471055874ull, 72057594037927936ull, 144396663052566528ull, 288794425616760832ull, 577588855528488960ull, 1155177711073755136ull, 2310355422147575808ull, 4620710844295151872ull, 9241421688590303745ull };

__constant__ static const uint64_t d2[] = { 1ull, 258ull, 66052ull, 16909320ull, 4328785936ull, 1108169199648ull, 283691315109952ull, 72624976668147840ull, 258ull, 66052ull, 16909320ull, 4328785936ull, 1108169199648ull, 283691315109952ull, 72624976668147840ull, 145249953336295424ull, 66052ull, 16909320ull, 4328785936ull, 1108169199648ull,
		283691315109952ull, 72624976668147840ull, 145249953336295424ull, 290499906672525312ull, 16909320ull, 4328785936ull, 1108169199648ull, 283691315109952ull, 72624976668147840ull, 145249953336295424ull, 290499906672525312ull, 580999813328273408ull, 4328785936ull, 1108169199648ull, 283691315109952ull, 72624976668147840ull,
		145249953336295424ull, 290499906672525312ull, 580999813328273408ull, 1161999622361579520ull, 1108169199648ull, 283691315109952ull, 72624976668147840ull, 145249953336295424ull, 290499906672525312ull, 580999813328273408ull, 1161999622361579520ull, 2323998145211531264ull, 283691315109952ull, 72624976668147840ull, 145249953336295424ull,
		290499906672525312ull, 580999813328273408ull, 1161999622361579520ull, 2323998145211531264ull, 4647714815446351872ull, 72624976668147840ull, 145249953336295424ull, 290499906672525312ull, 580999813328273408ull, 1161999622361579520ull, 2323998145211531264ull, 4647714815446351872ull, 9223372036854775808ull };

*/
