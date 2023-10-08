
#include "Cu_Common.h"

#include <numeric>
#include <iostream>
#include <random>
#include <chrono>
#include <numeric>
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
#include "cu_GeneticObstructionDiff.h"
#include "cu_Leorik.h"
#include "cu_SBAMG.h"
#include "cu_NoHeadache.h"
#include "cu_AVXShift.h"
#include "cu_SlideArithmInline.h"
#include "cu_Genetic8Ray.h"
#include "cu_Bitrotation.h"
#include "cu_foldingHash.h"
#include "cu_bitray.h"
#include "kernel.h"

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
        gpuErrchk(cudaMalloc(&attacks, threads * sizeof(uint64_t)));
        gpuErrchk(cudaMemset(attacks, 0, threads * sizeof(uint64_t)));
    }

    uint64_t MoveCount() {
        //return std::accumulate(counts, counts + threads, sum);
        return static_cast<uint64_t>(loops) * static_cast<uint64_t>(threads);
    }
};

template<int mode>
__device__ __inline__ uint64_t Queen(uint32_t sq, uint64_t occ) {
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
        return GeneticObstructionDifference::Queen(sq, occ);
    }
    if constexpr (mode == 13) {
        return Leorik::Queen(sq, occ);
    }
    if constexpr (mode == 14) {
        return SBAMG::Queen(sq, occ);
    }
    if constexpr (mode == 15) {
        return NOHEADACHE::Queen(sq, occ);
    }
    if constexpr (mode == 16) {
        return AVXShift::Queen(sq, occ);
    }
    if constexpr (mode == 17) {
        return SlideArithmInline::Queen(sq, occ);
    }
    if constexpr (mode == 18) {
        return Genetic8Ray::Queen(sq, occ);
    }
    if constexpr (mode == 19) {
        return Bitrotation::Queen(sq, occ);
    }
    if constexpr (mode == 20) {
        return FoldingHash::Queen(sq, occ);
    }
    if constexpr (mode == 21) {
        return Bitray::Queen(sq, occ);
    }
}
template<int mode>
__device__ __inline__ void Prepare(int threadIdx)
{
    if constexpr (mode == 2) BobLU::Prepare(threadIdx);
    if constexpr (mode == 4) HyperbolaQsc::Prepare(threadIdx);
    if constexpr (mode == 6) SlideArithm::Prepare(threadIdx);
    if constexpr (mode == 12) GeneticObstructionDifference::Prepare(threadIdx);
    if constexpr (mode == 18) Genetic8Ray::Prepare(threadIdx);
    if constexpr (mode == 19) Bitrotation::Prepare(threadIdx);
    if constexpr (mode == 20) FoldingHash::Prepare(threadIdx);
    if constexpr (mode == 21) Bitray::Prepare(threadIdx);

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
        case 9: return  "Hypercube Algorithm      ";
        case 10: return "Dumb 7 Fill              ";
        case 11: return "Obstruction Difference   ";
        case 12: return "Genetic Obstruction Diff ";
        case 13: return "Leorik                   ";
        case 14: return "SBAMG o^(o-3cbn)         ";
        case 15: return "NO HEADACHE              ";
        case 16: return "AVX Branchless Shift     ";
        case 17: return "Slide Arithmetic Inline  ";
        case 18: return "C++ Tree Sifter - 8 Rays ";
        case 19: return "Bitrotation o^(o-2r)     ";
        case 20: return "FoldingHash (uncomplete) ";
        case 21: return "Bitray 2023 version      ";
        default:
            return "";
    }
}

template<int mode>
__global__ void cu_GetQueenAttacks(Cuda_Chessprocessor params)
{
    int gid = getIdx();
    uint32_t x = 123456789 * gid, y = 362436069 ^ gid, z = 521288629 + (gid * gid + 1);
    Prepare<mode>(threadIdx.x);
    uint64_t* occs = params.attacks;
    uint64_t occmock = 0;
    const int loopcnt = params.loops;
    for (int i = 0; i < loopcnt; i++) {
        uint32_t sq = cu_rand32(x, y, z) & 63;
        uint64_t occ = cu_rand64(x, y, z);
        occmock ^= Queen<mode>(sq, occ);
    }
    occs[gid] = occmock;
}


template<int mode>
void TestChessprocessor(int blocks, int threadsperblock) {
    int lookups = blocks * threadsperblock;
    uint64_t nanoSeconds;
    uint64_t* results = new uint64_t[lookups];
    Cuda_Chessprocessor p(lookups);
    std::vector<double> avg;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << AlgoName(mode) << ":\t";

    //Default (sm52) default stream: 82GLU
    //Cuda Graph: 88GLU
    //Cuda Streams: 88GLU
    //Cuda compile settings optimisation: 114GLU
    //Optimize algorithm. New world record: 123 Billion Lookups/S for queens. RTX 3080 23.04.2022
    //Optimize bitrotation algorithm (horizontal ray). New world record: 142 Billion Lookups/S for queens. RTX 3080 28.04.2022
    {
        constexpr int streamcount = 8;
        cudaStream_t streams[streamcount];
        for (int i = 0; i < streamcount; i++)
        {
            gpuErrchk(cudaStreamCreate(streams + i));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 16; i++) 
        {
            cu_GetQueenAttacks<mode><<<blocks, threadsperblock,0, streams[i % streamcount]>>>(p);
            cudaVerifyLaunch();
        }
        for (int i = 0; i < streamcount; i++)
        {
            gpuErrchk(cudaStreamSynchronize(streams[i]));
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < streamcount; i++)
        {
            gpuErrchk(cudaStreamDestroy(streams[i]));
        }
        nanoSeconds = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    //Erase outliers
    //avg.erase(std::max_element(avg.begin(), avg.end()));
    //avg.erase(std::min_element(avg.begin(), avg.end()));

    double GigaLookups = p.MoveCount() * 16.0 / nanoSeconds;
    std::cout << GigaLookups << " GigaQueens/s\n";

    gpuErrchk(cudaFree(p.attacks));
    delete[] results;

}

void SetupDevice()
{
    gpuErrchk(cudaSetDevice(0));
    gpuErrchk(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    cudaDeviceProp prop;
    gpuErrchk(cudaGetDeviceProperties(&prop, 0));
    gpuErrchk(cudaDeviceSetSharedMemConfig(cudaSharedMemConfig::cudaSharedMemBankSizeEightByte));
    gpuErrchk(cudaDeviceSynchronize());
    std::cout << prop.name << "\n";
}

int main()
{
    SetupDevice();
    constexpr int blocks = 4096;
    constexpr int threadsperblock = 256;
    //TestCoprocessor(blocks, threadsperblock);
    //while (true) {
    //
    //    TestChessprocessor<19>(blocks, threadsperblock);
    //    return 0;
    //}
    //return;

    FancyHash::Init();
    Pext::Init();
    Hypercube::Init();
    SISSY::Init();

    gpuErrchk(cudaDeviceSynchronize());

    TestChessprocessor<0>(blocks, threadsperblock);
    TestChessprocessor<1>(blocks, threadsperblock);
    TestChessprocessor<2>(blocks, threadsperblock);
    TestChessprocessor<3>(blocks, threadsperblock);
    TestChessprocessor<4>(blocks, threadsperblock);
    TestChessprocessor<5>(blocks, threadsperblock);
    TestChessprocessor<6>(blocks, threadsperblock);
    TestChessprocessor<7>(blocks, threadsperblock);
    TestChessprocessor<8>(blocks, threadsperblock);
    //TestChessprocessor<9>(blocks, threadsperblock);
    TestChessprocessor<10>(blocks, threadsperblock);
    TestChessprocessor<11>(blocks, threadsperblock);
    TestChessprocessor<12>(blocks, threadsperblock);
    TestChessprocessor<13>(blocks, threadsperblock);
    TestChessprocessor<14>(blocks, threadsperblock);
    TestChessprocessor<15>(blocks, threadsperblock);
    TestChessprocessor<16>(blocks, threadsperblock);
    TestChessprocessor<17>(blocks, threadsperblock);
    TestChessprocessor<18>(blocks, threadsperblock);
    TestChessprocessor<19>(blocks, threadsperblock);
    TestChessprocessor<20>(blocks, threadsperblock);
    TestChessprocessor<21>(blocks, threadsperblock);
}