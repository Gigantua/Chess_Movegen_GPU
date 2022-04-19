#pragma once

//(c) Gerd Isenberg, Aleks Peshkov 2007
//Subtracting a Rook from a Blocking Piece - https://www.chessprogramming.org/Subtracting_a_Rook_from_a_Blocking_Piece

#include <stdint.h>
#include <array>
#include <type_traits>
#include "cu_Common.h"

//Cuda Translation by Daniel Inführ - Jan. 2022
//Contact: daniel.infuehr@live.de

namespace HyperbolaQsc {
	struct Mask {
		uint64_t diagonal;
		uint64_t antidiagonal;
		uint64_t vertical;
	};



	/* Init */
	static constexpr std::array<Mask, 64> InitMask() {
		int r{}, f{}, i{}, j{}, y{};
		int d[64]{};

		std::array<Mask, 64> MASK{};

		for (int x = 0; x < 64; ++x) {
			for (y = 0; y < 64; ++y) d[y] = 0;
			// directions
			for (i = -1; i <= 1; ++i)
				for (j = -1; j <= 1; ++j) {
					if (i == 0 && j == 0) continue;
					f = x & 07;
					r = x >> 3;
					for (r += i, f += j; 0 <= r && r < 8 && 0 <= f && f < 8; r += i, f += j) {
						y = 8 * r + f;
						d[y] = 8 * i + j;
					}
				}

			// uint64_t mask
			Mask& mask = MASK[x];
			for (y = x - 9; y >= 0 && d[y] == -9; y -= 9) mask.diagonal |= (1ull << y);
			for (y = x + 9; y < 64 && d[y] == 9; y += 9) mask.diagonal |= (1ull << y);

			for (y = x - 7; y >= 0 && d[y] == -7; y -= 7) mask.antidiagonal |= (1ull << y);
			for (y = x + 7; y < 64 && d[y] == 7; y += 7) mask.antidiagonal |= (1ull << y);

			for (y = x - 8; y >= 0; y -= 8) mask.vertical |= (1ull << y);
			for (y = x + 8; y < 64; y += 8) mask.vertical |= (1ull << y);
		}
		return MASK;
	}
	static constexpr std::array<uint8_t, 512> InitRank() {

		std::array<uint8_t, 512> rank_attack{};

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
	static const std::array<Mask, 64> init_mask = InitMask();
	static const std::array<uint8_t, 512> init_rank_attack = InitRank();

	__constant__ Mask mask[64];
	__constant__ uint8_t rank_attack[512];

	void Init() {
		gpuErrchk(cudaMemcpyToSymbol(mask, init_mask.data(), sizeof(init_mask)));
		gpuErrchk(cudaMemcpyToSymbol(rank_attack, init_rank_attack.data(), sizeof(init_rank_attack)));
	}

	__device__ uint64_t cu_bswap(uint64_t b) {
		b = ((b >> 8) & 0x00FF00FF00FF00FFULL) | ((b << 8) & 0xFF00FF00FF00FF00ULL);
		b = ((b >> 16) & 0x0000FFFF0000FFFFULL) | ((b << 16) & 0xFFFF0000FFFF0000ULL);
		b = ((b >> 32) & 0x00000000FFFFFFFFULL) | ((b << 32) & 0xFFFFFFFF00000000ULL);
		return b;
	}

	__device__ uint64_t bit_bswap(uint64_t b) {
		//Todo: check if __brevll can be used. (reverses bits not bytes)
		//return __brevll(b);
		return cu_bswap(b);
	}

	/* Generate attack using the hyperbola quintessence approach */
	__device__ uint64_t attack(uint64_t pieces, uint32_t x, uint64_t mask) {
		uint64_t o = pieces & mask;

		return ((o - (1ull << x)) ^ bit_bswap(bit_bswap(o) - (1ull << (x ^ 56)))) & mask;
	}

	__device__ uint64_t horizontal_attack(uint64_t pieces, uint32_t x) {
		uint32_t file_mask = x & 7;
		uint32_t rank_mask = x & 56;
		uint64_t o = (pieces >> rank_mask) & 126;

		return ((uint64_t)rank_attack[o * 4 + file_mask]) << rank_mask;
	}

	__device__ uint64_t vertical_attack(uint64_t pieces, uint32_t x) {
		return attack(pieces, x, mask[x].vertical);
	}

	__device__ uint64_t diagonal_attack(uint64_t pieces, uint32_t x) {
		return attack(pieces, x, mask[x].diagonal);
	}

	__device__ uint64_t antidiagonal_attack(uint64_t pieces, uint32_t x) {
		return attack(pieces, x, mask[x].antidiagonal);
	}

	__device__ uint64_t bishop_attack(int sq, uint64_t occ) {
		return diagonal_attack(occ, sq) | antidiagonal_attack(occ, sq);
	}

	__device__ uint64_t rook_attack(int sq, uint64_t occ) {
		return vertical_attack(occ, sq) | horizontal_attack(occ, sq);
	}

	__device__ uint64_t Queen(int sq, uint64_t occ) {
		return bishop_attack(sq, occ) | rook_attack(sq, occ);
	}
}