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

	constexpr std::array<uint8_t, 512> InitRank() {
		std::array<uint8_t, 512> rank_attack { };
		for (int x = 0; x < 64; ++x) {
			for (int f = 0; f < 8; ++f) {
				int o = 2 * x;
				int x2{}, y2{};
				int b{};

				y2 = 0;
				for (x2 = f - 1; x2 >= 0; --x2) {
					b = 1 << x2;
					y2 |= b;
					if ((o & b) == b) break;
				}
				for (x2 = f + 1; x2 < 8; ++x2) {
					b = 1 << x2;
					y2 |= b;
					if ((o & b) == b) break;
				}
				rank_attack[x * 8ull + f] = y2;
			}
		}
		return rank_attack;
	}

	template<uint64_t bb>
	constexpr BitFunction mask_shift(int ranks) {
		return ranks > 0 ? bb >> (ranks << 3) : bb << -(ranks << 3);
	}


	struct RayMask {
		uint64_t VE, D1, D2;
	};

	__shared__ RayMask shr[64];
	__shared__ uint8_t rank_attack[512];


	/* Generate attack using the hyperbola quintessence approach */
	/* ((occ - (1ull << sq)) ^ bit_reverse((bit_reverse(occ) - bit_reverse((1ull << sq))))) */
	BitFunction attack(uint64_t occ, uint64_t mask, uint64_t s_bit, uint64_t s_rev)
	{
		const uint64_t o = (occ & mask);
		return ((o - s_bit) ^ bitswap(bitswap(o) - s_rev)) & mask;
	}

	BitFunction horizontal_attack(uint64_t pieces, uint32_t x) {
		uint64_t o = (pieces >> (x & 56)) & 126;
		return rank_attack[o * 4 + (x & 7)] << (x & 56);
	}

	BitFunction Queen(int s, const uint64_t occ) {
		const uint64_t s_bit = 1ull << s;
		const uint64_t s_rev = (1ull << (s ^ 56));
		const auto& r = shr[s];
		return horizontal_attack(occ, s)
			 | (attack(occ, r.VE, s_bit, s_rev))
			 | (attack(occ, r.D1, s_bit, s_rev))
			 | (attack(occ, r.D2, s_bit, s_rev));
	}

	__inline__ __device__ void Prepare(unsigned int threadIdx)
	{
		if (threadIdx < 64) 
		{
			shr[threadIdx] = {
				dir_VE(threadIdx) ^ (1ull << threadIdx),
				dir_D1(threadIdx) ^ (1ull << threadIdx),
				dir_D2(threadIdx) ^ (1ull << threadIdx)
			};

			for (int i = 0; i < 8; i++)
			{
				int idx = 8 * threadIdx + i;
				rank_attack[idx] = InitRank()[idx];
			}
		}
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
