#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <mpi.h>
#include <cstdlib>
#include <chrono>

#include "jacobi.h"
#include "utils.h"

using namespace std::chrono;

template <typename T> void check_results(T* A, T* B, int nr, int nc)
{
    for (int i = 0; i < nr; i++)
    {
        for (int j = 0; j < nc; j++)
        {
            if (fabs(A[i*nc + j] - B[i*nc + j]) > 0.00001)
            {
                std::cerr << "ERROR AT IDX: ("<<i<<","<<j<<")" << std::endl;
                exit(0);
            }
        }
    }
    return;
}
void print_matrix(int nr, int nc, double* A) {
    int i,j;
    printf("Array is a %d x %d matrix\n\n",nr,nc);
    printf("%3s |","");
    for(i = 0; i < nc; i++) {
		printf("%4d ",i);
	}
	printf("\n%3s |","");
	for(j = 0; j < nc; j++) {
            printf("-----");
        }
    printf("\n");
    for(i = 0; i < nr; i++) {
        printf(" %2d |", i);
        for(j = 0; j < nc; j++) {
		printf("%4.2f ",A[i*nc+j]);
        }
        printf("\n");
    }
	printf("\n");

	return;
}

template<typename T>
T* create_matrix(int numRows, int numCols, float seed)
{
    //srand(seed);
    T* M = (T *)calloc(numRows * numCols, sizeof(T));

    for (int r = 0; r < numRows; r++)
    {
        for (int c = 0; c < numCols; c++)
        {
            if (r == 0 || r == numRows - 1) M[r * numCols + c] = 0.0;
            else if (c == 0 || c == numCols - 1) M[r * numCols + c] = 0.0;
            else M[r * numCols + c] = 10 * (rand() / (double)RAND_MAX);
        }
    }

    return M;
}

template <typename T>
void cpu_jacobi(T* M, T e, int iters, int numRows, int numCols)
{
    int k = 0;
    T num;
    T err = 1 + e;
    T *U = (T*)calloc(numRows * numCols, sizeof(T));

    while (k < iters && err > e)
    {
        err = 0.0;

        for (int i = 1; i < numRows - 1; i++)
        {
            for (int j = 1; j < numCols - 1; j++)
            {
                num = M[i * numCols + (j - 1)] 
                    + M[i * numCols + (j + 1)]
                    + M[(i - 1) * numCols + j]
                    + M[(i + 1) * numCols + j];
                U[i * numCols + j] = (num) / 4;                
                err += fabs(U[i * numCols + j] - M[i * numCols + j]);
            }
        }
        err = err / (numRows * numCols); 
        debug("Iter: %3d, Error: %0.10f\n", k, err);
        memcpy(M, U, numRows * numCols * sizeof(T));
        k++;
    }

    return;
}

int main(int argc, char const *argv[])
{
    int nr, nc, iter;
    double eb, *error_d;
    double *M_d, *U_d, *M_h, *U_h, *cu_out;
    nr = atoi(argv[1]);
    nc = atoi(argv[2]);
    iter = atoi(argv[3]);
    eb = atof(argv[4]);
    printf("%f\n", eb);

    M_h = create_matrix<double>(nr, nc, 5);
    U_h    = (double *)calloc(nr * nc, sizeof(double));
    cu_out = (double *)calloc(nr * nc, sizeof(double));

    //init cuda memory
    checkCudaErrors(cudaMalloc(&M_d, nr * nc * sizeof(double)));
    checkCudaErrors(cudaMalloc(&U_d, nr * nc * sizeof(double)));
    checkCudaErrors(cudaMalloc(&error_d, sizeof(double)));

    checkCudaErrors(cudaMemcpy(M_d, M_h, sizeof(double) * nr * nc, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(U_d, U_h, sizeof(double) * nr * nc, cudaMemcpyHostToDevice));

    // cpu version
    std::cout << "STARTING SERIAL..." <<std::endl;
    auto start = high_resolution_clock::now();
    cpu_jacobi<double>(M_h, eb, iter, nr, nc);
    auto stop = high_resolution_clock::now();

    auto cpu_duration = duration_cast<duration<double>>(stop - start);

    // cuda version
    
    std::cout << "STARTING CUDA..." <<std::endl;
    start = high_resolution_clock::now();
    launch_jacobi(M_d, U_d, error_d, eb, iter, nr, nc);
    stop = high_resolution_clock::now();

    auto cuda_duration = duration_cast<duration<double>>(stop - start);
    //copy results to host for checking
    checkCudaErrors(cudaMemcpy(cu_out, U_d, nc * nr * sizeof(double), cudaMemcpyDeviceToHost));

    //print_matrix(nr, nc, M_h);
    //print_matrix(nr, nc, cu_out);
    check_results(M_h, cu_out, nr, nc);

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::setprecision(6) << "SERIAL DURATION: " << cpu_duration.count() << std::endl;
    std::cout << std::setprecision(6) << "CUDA DURATION:   " << cuda_duration.count() << std::endl;

    free(M_h);
    free(U_h);
    free(cu_out);
    checkCudaErrors(cudaFree(M_d));
    checkCudaErrors(cudaFree(U_d));
    checkCudaErrors(cudaFree(error_d));
    return 0;
}