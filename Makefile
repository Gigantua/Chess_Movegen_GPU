CC=nvcc
CFLAGS=-gencode arch=compute_52,code=sm_52\
-gencode=arch=compute_60,code=sm_60\
-gencode=arch=compute_61,code=sm_61\
-gencode=arch=compute_70,code=sm_70\
-gencode=arch=compute_75,code=sm_75\
-gencode=arch=compute_80,code=sm_80\
-gencode=arch=compute_86,code=sm_86\
-gencode=arch=compute_87,code=sm_87\
--expt-relaxed-constexpr -std=c++17 --run --threads 8 -O3 kernel.cu
# -flto
default: all

all: kernel.cu
	$(CC) $(CFLAGS) -o movegen_gpu
	