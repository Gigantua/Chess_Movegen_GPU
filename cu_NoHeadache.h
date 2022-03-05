#pragma once

#include "cu_Common.h"

namespace NOHEADACHE 
{
    static constexpr uint64_t BB_AF = 0x0101010101010101;
    static constexpr uint64_t BB_BF = 0x0202020202020202;
    static constexpr uint64_t BB_CF = 0x0404040404040404;
    static constexpr uint64_t BB_DF = 0x0808080808080808;
    static constexpr uint64_t BB_EF = 0x1010101010101010;
    static constexpr uint64_t BB_FF = 0x2020202020202020;
    static constexpr uint64_t BB_GF = 0x4040404040404040;
    static constexpr uint64_t BB_HF = 0x8080808080808080;

    static constexpr uint64_t BB_R1 = 0x00000000000000FF;
    static constexpr uint64_t BB_R2 = 0x000000000000FF00;
    static constexpr uint64_t BB_R3 = 0x0000000000FF0000;
    static constexpr uint64_t BB_R4 = 0x00000000FF000000;
    static constexpr uint64_t BB_R5 = 0x000000FF00000000;
    static constexpr uint64_t BB_R6 = 0x0000FF0000000000;
    static constexpr uint64_t BB_R7 = 0x00FF000000000000;
    static constexpr uint64_t BB_R8 = 0xFF00000000000000;

    __device__ uint64_t Bishop(int s, uint64_t o)
    {
        uint64_t tmp, att = 0;
        o &= ~(1ull << s);
        tmp = 1ull << s; while ((tmp & (o | BB_HF | BB_R8)) == 0) { att |= tmp <<= 9; }
        tmp = 1ull << s; while ((tmp & (o | BB_AF | BB_R8)) == 0) { att |= tmp <<= 7; }
        tmp = 1ull << s; while ((tmp & (o | BB_AF | BB_R1)) == 0) { att |= tmp >>= 9; }
        tmp = 1ull << s; while ((tmp & (o | BB_HF | BB_R1)) == 0) { att |= tmp >>= 7; }
        return att;
    }

    __device__ uint64_t Rook(int s, uint64_t o)
    {
        uint64_t tmp, att = 0;
        o &= ~(1ull << s);
        tmp = 1ull << s; while ((tmp & (o | BB_HF)) == 0) { att |= tmp <<= 1; }
        tmp = 1ull << s; while ((tmp & (o | BB_AF)) == 0) { att |= tmp >>= 1; }
        tmp = 1ull << s; while ((tmp & (o | BB_R8)) == 0) { att |= tmp <<= 8; }
        tmp = 1ull << s; while ((tmp & (o | BB_R1)) == 0) { att |= tmp >>= 8; }
        return att;
    }
    __device__ uint64_t Queen(int s, uint64_t o)
    {
        return Bishop(s, o) | Rook(s, o);
    }
}


/* Daniel appendix: 
How to do fast lookups: 
__constant__
static const unsigned char BIT_MASK[4][8] = {
            { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 }, // 1 bit
            { 0x03, 0x0C, 0x30, 0xC0, 0x00, 0x00, 0x00, 0x00 }, // 2 bit
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, // Nan
            { 0x0F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }  // 4 bit
};

__global__ void j(unsigned char *d){
        *d = BIT_MASK[threadIdx.x][0];
}
*/