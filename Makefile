CC=clang++
CFLAGS= -O3 -mllvm -inline-threshold=16000 -march=native -funroll-loops -std=c++20 -pthread kernel.cu
# -flto
default: all

all: kernel.cu
	$(CC) $(CFLAGS) -o movegen_compare
	