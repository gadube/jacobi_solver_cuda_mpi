#include "jacobi.h"

#ifndef BLOCK
#define BLOCK 32
#endif

template <typename T>
__global__ void jacobi(T* U, T* M, T* error, size_t numRows, size_t numCols)
{
    T loc_err;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    T num;

    if ((j > 0) && (j < numCols - 1) && (i > 0) && (i < numRows - 1))
    {
        num = M[i * numCols + (j - 1)]
            + M[i * numCols + (j + 1)]
            + M[(i - 1) * numCols + j]
            + M[(i + 1) * numCols + j];
        U[i * numCols + j] = num / 4;
        loc_err = fabs(U[i * numCols + j] - M[i * numCols + j]);
        atomicAdd(error,loc_err);
    }
    return;
}

void launch_jacobi(double* M_d, double* U_d, double* error_d, double eb, int maxIter, int numRows, int numCols)
{
    double error_h;
    int k = 0;

    dim3 block(BLOCK, BLOCK, 1);
    dim3 grid((numCols + BLOCK - 1)/BLOCK, (numRows + BLOCK - 1)/BLOCK, 1);

    error_h = 1 + eb;

    while (k < maxIter && error_h > eb)
    {
        error_h = 0;
        checkCudaErrors(cudaMemcpy(error_d, &error_h, sizeof(double), cudaMemcpyHostToDevice));

        jacobi<double><<<grid,block>>>(U_d, M_d, error_d, numRows, numCols);
        //find_iter_error<double><<<grid,block>>>(U_d, M_d, error_d, eb, numRows, numCols);

        checkCudaErrors(cudaMemcpy(M_d, U_d, sizeof(double) * numRows * numCols, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(&error_h, error_d, sizeof(double), cudaMemcpyDeviceToHost));
        error_h = error_h / (numCols * numRows);
        debug("Iter: %3d, Error: %0.10f\n", k, error_h);
        k++; 
    }
}