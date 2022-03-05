#pragma once

//(c) Michael J Sherwin 2020
//SISSY BOARDS - https://www.chessprogramming.org/SISSY_Bitboards

//Cuda Translation by Daniel Inführ - Jan. 2022
//Contact: daniel.infuehr@live.de

#include <stdint.h>
#include "cu_Common.h"

namespace SISSY
{
    uint64_t qss[64][256][8];
    uint64_t bss[64][128][6];
    uint64_t bob[64];
    uint64_t b7e[64];
    uint64_t rob[64];

    __constant__ uint64_t* cu_qss_atk[1];
    __device__ uint64_t cu_qss(int x, int y, int z) {
        return (*cu_qss_atk)[x * 256 * 8 + y * 8 + z];
    }

    void InitializeBSS() {
        uint8_t sq, sqr, k, l;
        int8_t x, dx, y, dy;
        uint64_t b, bb, i, j, ii, jj;

        for (sq = 0; sq < 64; sq++) {
            y = sq >> 3;
            x = sq & 7;
            for (i = 0; i < 128; i++) {
                for (k = 8, l = 0; k <= 48; k += 8, l++) {
                    bb = 0;
                    b = ((uint64_t)i << k) & bob[sq];
                    for (dx = +1, dy = +1; x + dx < +8 && y + dy < +8; dx++, dy++) {
                        sqr = (((y + dy) << 3) + x + dx);
                        bb |= 1ull << sqr;
                        if ((1ull << sqr) & b) break;
                    }
                    for (dx = -1, dy = +1; x + dx > -1 && y + dy < +8; dx--, dy++) {
                        sqr = (((y + dy) << 3) + x + dx);
                        bb |= 1ull << sqr;
                        if ((1ull << sqr) & b) break;
                    }
                    for (dx = +1, dy = -1; x + dx < +8 && y + dy > -1; dx++, dy--) {
                        sqr = (((y + dy) << 3) + x + dx);
                        bb |= 1ull << sqr;
                        if ((1ull << sqr) & b) break;
                    }
                    for (dx = -1, dy = -1; x + dx > -1 && y + dy > -1; dx--, dy--) {
                        sqr = (((y + dy) << 3) + x + dx);
                        bb |= 1ull << sqr;
                        if ((1ull << sqr) & b) break;
                    }
                    bss[sq][i][l] = bb;
                }
            }
        }

        for (sq = 0; sq <= 63; sq++) {
            for (ii = 0; ii <= 127; ii++) {
                for (jj = 0; jj <= 127; jj++) {
                    i = ((ii << 8) & bob[sq]) >> 8;
                    j = ((jj << 32) & bob[sq]) >> 32;
                    bss[sq][i | j][0] = bss[sq][i][0] & bss[sq][j][3];
                    i = ((ii << 16) & bob[sq]) >> 16;
                    j = ((jj << 40) & bob[sq]) >> 40;
                    bss[sq][i | j][1] = bss[sq][i][1] & bss[sq][j][4];
                    i = ((ii << 24) & bob[sq]) >> 24;
                    j = ((jj << 48) & bob[sq]) >> 48;
                    bss[sq][i | j][2] = bss[sq][i][2] & bss[sq][j][5];
                }
            }
        }
    }
    void InitializeQSS() {
        uint8_t sq, sqr, k, l;
        int8_t x, y, dx, dy;
        int32_t i;
        uint64_t b, bb;

        for (sq = 0; sq < 64; sq++) {
            y = sq >> 3;
            x = sq & 7;
            bob[sq] = 0;
            rob[sq] = 0;
            for (i = 0; i < 256; i++) {
                for (k = 0, l = 0; k <= 56; k += 8, l++) {
                    bb = 0;
                    b = (uint64_t)i << k;
                    for (dx = +1, dy = +1; x + dx < +8 && y + dy < +8; dx++, dy++) {
                        sqr = (((y + dy) << 3) + x + dx);
                        bb |= 1ull << sqr;
                        bob[sq] |= 1ull << sqr;
                        if ((1ull << sqr) & b) break;
                    }
                    for (dx = -1, dy = +1; x + dx > -1 && y + dy < +8; dx--, dy++) {
                        sqr = (((y + dy) << 3) + x + dx);
                        bb |= 1ull << sqr;
                        bob[sq] |= 1ull << sqr;
                        if ((1ull << sqr) & b) break;
                    }
                    for (dx = +1, dy = -1; x + dx < +8 && y + dy > -1; dx++, dy--) {
                        sqr = (((y + dy) << 3) + x + dx);
                        bb |= 1ull << sqr;
                        bob[sq] |= 1ull << sqr;
                        if ((1ull << sqr) & b) break;
                    }
                    for (dx = -1, dy = -1; x + dx > -1 && y + dy > -1; dx--, dy--) {
                        sqr = (((y + dy) << 3) + x + dx);
                        bb |= 1ull << sqr;
                        bob[sq] |= 1ull << sqr;
                        if ((1ull << sqr) & b) break;
                    }
                    for (dx = -1; x + dx > -1; dx--) {
                        sqr = (y << 3) + x + dx;
                        bb |= 1ull << sqr;
                        rob[sq] |= 1ull << sqr;
                        if ((1ull << sqr) & b) break;
                    }
                    for (dx = +1; x + dx < +8; dx++) {
                        sqr = (y << 3) + x + dx;
                        bb |= 1ull << sqr;
                        rob[sq] |= 1ull << sqr;
                        if ((1ull << sqr) & b) break;
                    }
                    for (dy = +1; y + dy < +8; dy++) {
                        sqr = ((y + dy) << 3) + x;
                        bb |= 1ull << sqr;
                        rob[sq] |= 1ull << sqr;
                        if ((1ull << sqr) & b) break;
                    }
                    for (dy = -1; y + dy > -1; dy--) {
                        sqr = ((y + dy) << 3) + x;
                        bb |= 1ull << sqr;
                        rob[sq] |= 1ull << sqr;
                        if ((1ull << sqr) & b) break;
                    }
                    qss[sq][i][l] = bb;
                }
            }
            b7e[sq] = bob[sq] & 0x007e7e7e7e7e7e00;
        }
    }

    void Init() {
        InitializeQSS();
        InitializeBSS();

        uint64_t* attacks;
        gpuErrchk(cudaMalloc(&attacks, sizeof(qss)));
        gpuErrchk(cudaMemcpyToSymbol(cu_qss_atk, &attacks, sizeof(uint64_t*)));

        for (int x = 0; x < 64; x++) {
            for (int y = 0; y < 256; y++) {
                int offset = x * 256 * 8 + y * 8;
                int len = 8 * sizeof(uint64_t);
                gpuErrchk(cudaMemcpy(attacks + offset, qss[x][y], len, cudaMemcpyKind::cudaMemcpyHostToDevice));
            }
        }
    }

    __device__ uint64_t Queen(int sq, uint64_t occ) {
        return cu_qss(sq, occ & 255        ,0)
             & cu_qss(sq, (occ >> 8) & 255 ,1)
             & cu_qss(sq, (occ >> 16) & 255,2)
             & cu_qss(sq, (occ >> 24) & 255,3)
             & cu_qss(sq, (occ >> 32) & 255,4)
             & cu_qss(sq, (occ >> 40) & 255,5)
             & cu_qss(sq, (occ >> 48) & 255,6)
             & cu_qss(sq, (occ >> 56) & 255,7);
    }
}