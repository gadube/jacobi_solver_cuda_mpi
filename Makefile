CC=g++
MCC=mpicxx
NVCC=nvcc

DBG=-DDEBUG

CUDA_INCLUDEPATH=/usr/local/cuda/include

NVCC_OPTS=-arch=sm_70 -DBLOCK=32 $(DBG)

GCC_OPTS=-std=c++11 -g -O3 -Wall $(DBG)
CUDA_LD_FLAGS=-L $(CUDA_ROOT)/lib64 -lcudart

final: main.o solver.o jacobi.h utils.h
	$(MCC) -o jacobi main.o jacobi.o $(CUDA_LD_FLAGS)

main.o: main.cpp jacobi.h utils.h 
	$(MCC) -c $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) main.cpp 

solver.o: jacobi.cu jacobi.h utils.h
	$(NVCC) -c jacobi.cu $(NVCC_OPTS)

clean:
	rm -rf *.o jacobi
