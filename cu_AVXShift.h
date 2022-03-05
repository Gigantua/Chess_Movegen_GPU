#pragma once

// Datum  : 10.02.2022
// Author : (c) Daniel Infuehr
// Content: AVX2 Branchless Fill Algorithm
// This header must be included without any change if you use parts of this code

#pragma once
#include <stdint.h>
#include <algorithm>

#include "cu_Common.h"

namespace AVXShift
{
    static constexpr uint64_t BB_AF = 0x0101010101010101;
    static constexpr uint64_t BB_HF = 0x8080808080808080;
    static constexpr uint64_t BB_R1 = 0x00000000000000FF;
    static constexpr uint64_t BB_R8 = 0xFF00000000000000;

    struct Vec4I {
        ulonglong4 ymm;

        __device__ Vec4I()
        { } //Empty Constructor

        __device__ Vec4I(ulonglong4&& value) : ymm(value)
        {

        }

        __device__ Vec4I(uint64_t value) : ymm(make_ulonglong4(value, value, value, value))
        {

        }
        __device__ Vec4I(uint64_t a, uint64_t b, uint64_t c, uint64_t d) : ymm(make_ulonglong4(a, b, c, d))
        {

        }

        inline __device__ Vec4I operator|(const Vec4I& rhs) const {
            return Vec4I(ymm.x | rhs.ymm.x, ymm.y | rhs.ymm.y, ymm.z | rhs.ymm.z, ymm.w | rhs.ymm.w);
        }

        inline __device__ Vec4I& operator|=(const Vec4I& rhs) {
            ymm.x |= rhs.ymm.x;
            ymm.y |= rhs.ymm.y;
            ymm.z |= rhs.ymm.z;
            ymm.w |= rhs.ymm.w;
            return *this;
        }

        inline __device__ Vec4I& operator&=(const Vec4I& rhs) {
            ymm.x &= rhs.ymm.x;
            ymm.y &= rhs.ymm.y;
            ymm.z &= rhs.ymm.z;
            ymm.w &= rhs.ymm.w;
            return *this;
        }

        inline __device__ Vec4I& operator<<=(const Vec4I& rhs) {
            ymm.x <<= rhs.ymm.x;
            ymm.y <<= rhs.ymm.y;
            ymm.z <<= rhs.ymm.z;
            ymm.w <<= rhs.ymm.w;
            return *this;
        }

        inline __device__ Vec4I& operator>>=(const Vec4I& rhs) {
            ymm.x >>= rhs.ymm.x;
            ymm.y >>= rhs.ymm.y;
            ymm.z >>= rhs.ymm.z;
            ymm.w >>= rhs.ymm.w;
            return *this;
        }

        inline __device__ Vec4I& andNot(const Vec4I& rhs) {
            ymm.x &= ~rhs.ymm.x;
            ymm.y &= ~rhs.ymm.y;
            ymm.z &= ~rhs.ymm.z;
            ymm.w &= ~rhs.ymm.w;
            return *this;
        }

        inline __device__ void Set(const Vec4I& rhs)
        {
            ymm = rhs.ymm;
        }

        inline __device__ void Set(uint64_t a, uint64_t b, uint64_t c, uint64_t d)
        {
            ymm = make_ulonglong4(a, b, c, d);
        }
        inline __device__ void Set(uint64_t a)
        {
            ymm = make_ulonglong4(a,a,a,a);
        }

        inline __device__ void Zero() {
            ymm = make_ulonglong4(0,0,0,0);
        }

        inline __device__ uint64_t horizontal_or() const {
            return ymm.x | ymm.y | ymm.z | ymm.w;
        }
    };

    static inline __device__ uint64_t Queen(const int s, uint64_t o)
    {
        //This is branchless improvement of 'NO HEADACHES' algorithm code. (no stop condition. Expand 7 times) 
        
        const Vec4I shift = Vec4I(9, 7, 1, 8);
        const Vec4I Occ = Vec4I(o).andNot(1ull << s);
        Vec4I A, tmp, att;
        att.Zero();

        //Same code repeated 7x per 4 rays: 
        //att.ymm = _mm256_or_si256(att.ymm, tmp.ymm = _mm256_sllv_epi64(_mm256_andnot_si256(A.ymm, tmp.ymm), shift.ymm));

        A = (Occ | Vec4I(BB_HF | BB_R8, BB_AF | BB_R8, BB_HF, BB_R8));
        tmp = (1ull << s);
        tmp.andNot(A); att |= tmp <<= shift;
        tmp.andNot(A); att |= tmp <<= shift;
        tmp.andNot(A); att |= tmp <<= shift;
        tmp.andNot(A); att |= tmp <<= shift;
        tmp.andNot(A); att |= tmp <<= shift;
        tmp.andNot(A); att |= tmp <<= shift;
        tmp.andNot(A); att |= tmp <<= shift;

        A = (Occ | Vec4I(BB_AF | BB_R1, BB_HF | BB_R1, BB_AF, BB_R1));
        tmp = (1ull << s);
        tmp.andNot(A); att |= tmp >>= shift;
        tmp.andNot(A); att |= tmp >>= shift;
        tmp.andNot(A); att |= tmp >>= shift;
        tmp.andNot(A); att |= tmp >>= shift;
        tmp.andNot(A); att |= tmp >>= shift;
        tmp.andNot(A); att |= tmp >>= shift;
        tmp.andNot(A); att |= tmp >>= shift;

        return att.horizontal_or();
    }
}
