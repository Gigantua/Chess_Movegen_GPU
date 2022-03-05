#pragma once
#include "cu_Common.h"
#include <mma.h>
using namespace nvcuda::wmma;


__device__ void tensor_op_16_16_16(
	int* d, void* a, void* b, float* c)
{
	fragment<matrix_a, 8, 8, 128, experimental::precision::b1, row_major> Amat;
	fragment<matrix_b, 8, 8, 128, experimental::precision::b1, col_major> Bmat;
	fragment<accumulator, 8, 8, 128, int> Cmat;
	
	load_matrix_sync(Amat, a, 16);
	load_matrix_sync(Bmat, b, 16);
	fill_fragment(Cmat, 0.0f);
	bmma_sync(Cmat, Amat, Bmat, Cmat);
	store_matrix_sync(d, Cmat, 16, layout_t::mem_col_major);
}