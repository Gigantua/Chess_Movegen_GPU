
#include "Cu_Common.h"

#include <numeric>
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include "cu_Bob.h"
#include "cu_FancyHash.h"
#include "cu_QBB.h"
#include "cu_KoggeStone.h"
#include "cu_Hyperbola.h"
#include "cu_Switch.h"
#include "cu_Pext.h"
#include "cu_SlideArithm.h"
#include "cu_Sissy.h"
#include "cu_Hypercube.h"
#include "cu_Dumb7Fill.h"
#include "cu_ObstructionDiff.h"
#include "cu_Leorik.h"
#include "cu_SBAMG.h"
#include "cu_NoHeadache.h"
#include "cu_AVXShift.h"
#include "cu_SlideArithmInline.h"
#include "cu_Bitrotation.h"

/// <summary>
/// Complete Algorithm Runs on the GPU only
/// </summary>
struct Cuda_Chessprocessor
{
    uint64_t* attacks;
    long threads;
    int loops = 256;

    //Output of iterations each thread did perform
    Cuda_Chessprocessor(long threads) : threads(threads)
    {
        gpuErrchk(cudaSetDevice(0));
        gpuErrchk(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaMalloc(&attacks, threads * sizeof(uint64_t)));
        //gpuErrchk(cudaHostAlloc(&attacks, threads * sizeof(uint64_t), cudaHostAllocMapped));
    }

    uint64_t LookupCount() {
        //return std::accumulate(counts, counts + threads, sum);
        return static_cast<uint64_t>(loops) * static_cast<uint64_t>(threads);
    }
};

template<int mode>
__device__ __inline__ uint64_t Queen(uint32_t sq, uint64_t occ, uint32_t& x, uint32_t& y, uint32_t& z) {
    if constexpr (mode == 0) {
        return FancyHash::Queen(sq, occ);
    }
    if constexpr (mode == 1) {
        return QBB::Queen(sq, occ);
    }
    if constexpr (mode == 2) {
        return BobLU::Queen(sq, occ);
    }
    if constexpr (mode == 3) {
        return KoggeStone::Queen(sq, occ);
    }
    if constexpr (mode == 4) {
        return HyperbolaQsc::Queen(sq, occ);
    }
    if constexpr (mode == 5) {
        return SwitchLookup::Queen(sq, occ);
    }
    if constexpr (mode == 6) {
        return SlideArithm::Queen(sq, occ);
    }
    if constexpr (mode == 7) {
        return Pext::Queen(sq, occ);
    }
    if constexpr (mode == 8) {
        return SISSY::Queen(sq, occ);
    }
    if constexpr (mode == 9) {
        return Hypercube::Queen(sq, occ);
    }
    if constexpr (mode == 10) {
        return Dumb7Fill::Queen(sq, occ);
    }
    if constexpr (mode == 11) {
        return ObstructionDifference::Queen(sq, occ);
    }
    if constexpr (mode == 12) {
        return Leorik::Queen(sq, occ);
    }
    if constexpr (mode == 13) {
        return SBAMG::Queen(sq, occ);
    }
    if constexpr (mode == 14) {
        return NOHEADACHE::Queen(sq, occ);
    }
    if constexpr (mode == 15) {
        return AVXShift::Queen(sq, occ);
    }
    if constexpr (mode == 16) {
        return SlideArithmInline::Queen(sq, occ);
    }
    if constexpr (mode == 17) {
        return Bitrotation::Queen(sq, occ);
    }
}
const char* AlgoName(int mode) {
    switch (mode)
    {
        case 0: return  "Black Magic - Fixed shift";
        case 1: return  "QBB Algo                 ";
        case 2: return  "Bob Lookup               ";
        case 3: return  "Kogge Stone              ";
        case 4: return  "Hyperbola Quiescence     ";
        case 5: return  "Switch Lookup            ";
        case 6: return  "Slide Arithm             ";
        case 7: return  "Pext Lookup              ";
        case 8: return  "SISSY Lookup             ";
        case 9: return  "Hypercube Alg            ";
        case 10: return "Dumb 7 Fill              ";
        case 11: return "Obstruction Difference   ";
        case 12: return "Leorik                   ";
        case 13: return "SBAMG o^(o-3cbn)         ";
        case 14: return "NO HEADACHE              ";
        case 15: return "AVX Branchless Shift     ";
        case 16: return "Slide Arithmetic Inline  ";
        case 17: return "Bitrotation o^(o-2r)     ";
        default:
            return "";
    }
}

template<int mode>
__global__ void cu_GetQueenAttacks(Cuda_Chessprocessor params)
{
    const int gid = getIdx();
    uint32_t x = 123456789 * gid, y = 362436069 ^ gid, z = 521288629 + (gid * gid + 1);

    volatile uint64_t* occs = params.attacks;
    const int loopcnt = params.loops;
    for (int i = 0; i < loopcnt; i++) {
        uint32_t sq = cu_rand32(x, y, z) & 63;
        uint64_t occ = cu_rand64(x, y, z);
        occs[gid] ^= Queen<mode>(sq, occ, x, y, z);
    }
}

template<int mode>
void TestChessprocessor(int blocks, int threadsperblock) {
    int lookups = blocks * threadsperblock;
    uint64_t* results = new uint64_t[lookups];
    Cuda_Chessprocessor p(lookups);
    std::vector<double> avg;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << AlgoName(mode) << ":\t";
    for (int i = 0; i < 12; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        cu_GetQueenAttacks<mode><<<blocks, threadsperblock >>>(p);

        auto err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf(cudaGetErrorString(err));
            exit(11);
        }
        gpuErrchk(cudaDeviceSynchronize());
        auto t2 = std::chrono::high_resolution_clock::now();

        //verification of buffer on the cpu side. 
        //cudaMemcpy(results, p.attacks, lookups * 8, cudaMemcpyKind::cudaMemcpyDeviceToHost);

        double GLU = p.LookupCount() * 1.0 / std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        avg.push_back(GLU);
        //std::cout << "\t" << MLU << "\n";
    }
    //Erase outliers
    avg.erase(std::max_element(avg.begin(), avg.end()));
    avg.erase(std::min_element(avg.begin(), avg.end()));

    double MegaLookups = accumulate(avg.begin(), avg.end(), 0.0) / avg.size();
    std::cout << MegaLookups << " GigaQueens/s\n";
    delete[] results;
}


int main()
{
    constexpr int blocks = 4096;
    constexpr int threadsperblock = 256;
    //TestCoprocessor(blocks, threadsperblock);
    cudaDeviceProp prop;
    gpuErrchk(cudaGetDeviceProperties(&prop, 0));
    std::cout << prop.name << "\n"; 

    BobLU::Init();
    HyperbolaQsc::Init();
    FancyHash::Init();
    Pext::Init();
    Hypercube::Init();
    SISSY::Init();

    //Leorik, QBB, ObstructionDiff

    TestChessprocessor<17>(blocks, threadsperblock);
    TestChessprocessor<17>(blocks, threadsperblock);
    TestChessprocessor<0>(blocks, threadsperblock);
    TestChessprocessor<1>(blocks, threadsperblock);
    TestChessprocessor<2>(blocks, threadsperblock);
    TestChessprocessor<3>(blocks, threadsperblock);
    TestChessprocessor<4>(blocks, threadsperblock);
    TestChessprocessor<5>(blocks, threadsperblock);
    TestChessprocessor<6>(blocks, threadsperblock);
    TestChessprocessor<7>(blocks, threadsperblock);
    TestChessprocessor<8>(blocks, threadsperblock);
    TestChessprocessor<9>(blocks, threadsperblock);
    TestChessprocessor<10>(blocks, threadsperblock);
    TestChessprocessor<11>(blocks, threadsperblock);
    TestChessprocessor<12>(blocks, threadsperblock);
    TestChessprocessor<13>(blocks, threadsperblock);
    TestChessprocessor<14>(blocks, threadsperblock);
    TestChessprocessor<15>(blocks, threadsperblock);
    TestChessprocessor<16>(blocks, threadsperblock);
    TestChessprocessor<17>(blocks, threadsperblock);
}