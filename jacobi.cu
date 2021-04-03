#include "jacobi.h"

#ifndef BLOCK
#define BLOCK 32
#endif

template <typename T>
__global__ void jacobi(T* U, T* M, T* error, size_t numRows, size_t numCols)
{
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
        error[i * numCols + j] = fabs(U[i * numCols + j] - M[i * numCols + j]);
    }
    return;
}

__global__
void find_sum(double* M, int size, double* out)
{
    __shared__ double s_M[1024];

    int tid = threadIdx.x;
    int i = blockIdx.x * 1024 + threadIdx.x;
    int gridSz = 1024 * gridDim.x;
    double total = 0;

    for (int idx = i; i < size; i += gridSz)
    {
        total += M[idx];
    }

    s_M[tid] = total;
    __syncthreads();

    for (int len = 1024 / 2; len > 0; len = len / 2)
    {
        if (tid < len)
        {
            s_M[tid] += s_M[tid + len];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = s_M[0];
    }




}

void launch_jacobi(double* M_d, double* U_d, double* error_d, double* toterr, double eb, int maxIter, int numRows, int numCols)
{
    int rank, k = 0;
    double error;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    dim3 block(BLOCK, BLOCK, 1);
    dim3 grid((numCols + BLOCK - 1)/BLOCK, (numRows + BLOCK - 1)/BLOCK, 1);

    do
    {
        error = 0.0;
        jacobi<double><<<grid,block>>>(U_d, M_d, error_d, numRows, numCols);
        checkCudaErrors(cudaMemcpy(M_d, U_d, sizeof(double) * numRows * numCols, cudaMemcpyDeviceToDevice));

        find_sum<<<24,1024>>>(error_d, numRows * numCols, toterr);
        find_sum<<<1,1024>>>(toterr, 24, toterr);
        checkCudaErrors(cudaMemcpy(&error, toterr, sizeof(double), cudaMemcpyDeviceToHost));
        error = error / (numCols * numRows);
        debug("%d | Iter: %3d, Error: %0.10f\n", rank, k, error);
        k++; 
    }while (k < maxIter && error > eb);
}
