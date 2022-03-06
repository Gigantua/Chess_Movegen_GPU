#pragma once

//(c) Gerd Isenberg, Aleks Peshkov 2007
//Subtracting a Rook from a Blocking Piece - https://www.chessprogramming.org/Subtracting_a_Rook_from_a_Blocking_Piece

#include <stdint.h>
#include <array>
#include <type_traits>
#include "cu_Common.h"

//Cuda Translation by Daniel Inführ - Jan. 2022
//Contact: daniel.infuehr@live.de

namespace Bitrotation {
#define BitFunction __inline__ __device__ uint64_t

	template<uint64_t bb>
	BitFunction mask_shift(int ranks) {
		return ranks > 0 ? bb >> (ranks << 3) : bb << -(ranks << 3);
	}

#	define dir_HO(X) (0xFFull << (X & 56))
#	define dir_VE(X) (0x0101010101010101ull << (X & 7))
#	define dir_D1(X) (mask_shift<0x8040201008040201ull>((X & 7) - (X >> 3)))
#	define dir_D2(X) (mask_shift<0x0102040810204080ull>(7 - (X & 7) - (X >> 3)))

	/* Generate attack using the hyperbola quintessence approach */
	BitFunction attack(uint64_t o, uint32_t sq) 
	{
		return ((o - (1ull << sq)) ^ __brevll(__brevll(o) - (1ull << (sq ^ 63))));
	}

	BitFunction Queen(const int s, const uint64_t occ) {
		return (attack(occ & dir_HO(s), s) & dir_HO(s))
			 ^ (attack(occ & dir_VE(s), s) & dir_VE(s))
			 ^ (attack(occ & dir_D1(s), s) & dir_D1(s))
			 ^ (attack(occ & dir_D2(s), s) & dir_D2(s));
	}
#undef BitFunction
}