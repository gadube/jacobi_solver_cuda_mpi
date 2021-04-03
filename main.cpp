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
                MPI_Finalize();
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
void cpu_jacobi(T* M, T e, int* iters, int numRows, int numCols)
{
    int k = 0;
    T num;
    T err = 1 + e;
    T *U = (T*)calloc(numRows * numCols, sizeof(T));

    while (k < *iters && err > e)
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
    //*iters = k;

    return;
}

double* add_column(double* Old, double* vector, int side, int rows, int* cols)
{
    double* New = (double *)calloc(rows * (*cols + 1), sizeof(double));

    if (side == 0) //left
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < *cols; j++)
            {
                New[i * (*cols + 1) + j + 1] = Old[i * (*cols) + j];
            }
            New[i * (*cols + 1)] = vector[i];
        }
    }
    else if (side == 1) //right
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < *cols; j++)
            {
                New[i * (*cols + 1) + j] = Old[i * (*cols) + j];
            }
            New[i * (*cols + 1) + *cols] = vector[i];
        }
    }

    (*cols)++;
    free(Old);
    return New;
}

double* add_row(double* Old, double* vector, int side, int* rows, int cols)
{
    double* New = (double *)calloc((*rows + 1) * cols, sizeof(double));

    if (side == 0) //up
    {
        for (int i = 0; i < *rows + 1; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i == 0) New[i * cols + j] = vector[j];
                else New[i * (cols) + j] = Old[(i-1) * cols + j];
            }
        }
    }
    else if (side == 1) //down
    {
        for (int i = 0; i < *rows + 1; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i == *rows) New[i * cols + j] = vector[j];
                else New[i * (cols) + j] = Old[i * cols + j];
            }
        }
    }

    (*rows)++;
    free(Old);
    return New;
}

double* comm_columns(double* locM, int* rows, int* cols, MPI_Comm cart_comm, int left, int right)
{
    int size, rank;
    MPI_Status status;
    double* newM = NULL;
    MPI_Comm_size(cart_comm, &size);
    MPI_Comm_rank(cart_comm, &rank);

    // neighbor vals to recv
    double* nl = (double *)calloc(*rows, sizeof(double));
    double* nr = (double *)calloc(*rows, sizeof(double));

    // local vals to send
    double* ll = (double *)calloc(*rows, sizeof(double));
    double* lr = (double *)calloc(*rows, sizeof(double));

    // fill local buffers
    for (int i = 0; i < *rows; i++)
    {
        ll[i] = locM[i * (*cols)];
        lr[i] = locM[(i+1) * (*cols) - 1];
    }
    if (left >= 0 && right >= 0)
    {
        MPI_Sendrecv(ll, *rows, MPI_DOUBLE, left, 0, nl, *rows, MPI_DOUBLE, right, 0, cart_comm, &status);
        MPI_Sendrecv(lr, *rows, MPI_DOUBLE, right, 0, nr, *rows, MPI_DOUBLE, left, 0, cart_comm, &status);
        newM = add_column(locM, nl, 1, *rows, cols); 
        newM = add_column(newM, nr, 0, *rows, cols); 
    for (int proc=0; proc<size; proc++) {
        if (proc == rank) {
            printf("Rank = %d, RL = %d, %d\n", rank, right, left);
            printf("Local Matrix:\n");
            for (int ii=0; ii<*rows; ii++) {
                for (int jj=0; jj<*cols; jj++) {
                    printf("%3.1f ",newM[ii*(*cols)+jj]);
                }
                printf("\n");
            }
            printf("\n");

            printf("Nbr Left | Nbr Right:\n");
            for (int ii=0; ii < *rows; ii++)
            {
                printf("%3.1f | %3.1f\n",nl[ii], nr[ii]);
            }
            printf("\n");
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    }
    else if (left >= 0)
    {
        MPI_Sendrecv(ll, *rows, MPI_DOUBLE, left, 0, nr, *rows, MPI_DOUBLE, left, 0, cart_comm, &status);
        newM = add_column(locM, nr, 0, *rows, cols); 
    for (int proc=0; proc<size; proc++) {
        if (proc == rank) {
            printf("Rank = %d, RL = %d, %d\n", rank, right, left);
            printf("Local Matrix:\n");
            for (int ii=0; ii<*rows; ii++) {
                for (int jj=0; jj<*cols; jj++) {
                    printf("%3.1f ",newM[ii*(*cols)+jj]);
                }
                printf("\n");
            }
            printf("\n");

            printf("Local Left | Nbr Right:\n");
            for (int ii=0; ii < *rows; ii++)
            {
                printf("%3.1f | %3.1f\n",ll[ii], nr[ii]);
            }
            printf("\n");
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    }
    else if (right >= 0)
    {
        MPI_Sendrecv(lr, *rows, MPI_DOUBLE, right, 0, nl, *rows, MPI_DOUBLE, right, 0, cart_comm, &status);
        newM = add_column(locM, nl, 1, *rows, cols); 
    for (int proc=0; proc<size; proc++) {
        if (proc == rank) {
            printf("Rank = %d, RL = %d, %d\n", rank, right, left);
            printf("Local Matrix:\n");
            for (int ii=0; ii<*rows; ii++) {
                for (int jj=0; jj<*cols; jj++) {
                    printf("%3.1f ",newM[ii*(*cols)+jj]);
                }
                printf("\n");
            }
            printf("\n");

            printf("Nbr Left | Local Right:\n");
            for (int ii=0; ii < *rows; ii++)
            {
                printf("%3.1f | %3.1f\n",nl[ii], lr[ii]);
            }
            printf("\n");
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    }

    free(nl);
    free(nr);
    free(ll);
    free(lr);
    return newM;
}

double* comm_rows(double* locM, int* rows, int* cols, MPI_Comm cart_comm, int down, int up)
{
    int size, rank;
    double* newM = NULL;
    MPI_Status status;
    MPI_Comm_size(cart_comm, &size);
    MPI_Comm_rank(cart_comm, &rank);

    // neighbor vals to recv
    double* nu = (double *)calloc(*cols, sizeof(double));
    double* nd = (double *)calloc(*cols, sizeof(double));

    // local vals to send
    double* lu = (double *)calloc(*cols, sizeof(double));
    double* ld = (double *)calloc(*cols, sizeof(double));

    // fill local buffers
    for (int i = 0; i < *cols; i++)
    {
        lu[i] = locM[i];
        ld[i] = locM[(*rows - 1) * (*cols) + i];
    }
    if (down >= 0 && up >= 0)
    {
        MPI_Sendrecv(lu, *cols, MPI_DOUBLE, up, 0, nu, *cols, MPI_DOUBLE, down, 0, cart_comm, &status);
        MPI_Sendrecv(ld, *cols, MPI_DOUBLE, down, 0, nd, *cols, MPI_DOUBLE, up, 0, cart_comm, &status);
        newM = add_row(locM, nu, 1, rows, *cols); 
        newM = add_row(newM, nd, 0, rows, *cols); 
    for (int proc=0; proc<size; proc++) {
        if (proc == rank) {
            printf("Rank = %d, UD = %d, %d\n", rank, up, down);
            printf("Local Matrix:\n");
            for (int ii=0; ii<*rows; ii++) {
                for (int jj=0; jj<*cols; jj++) {
                    printf("%3.1f ",newM[ii*(*cols)+jj]);
                }
                printf("\n");
            }
            printf("\n");

            printf("Nbr Up | Nbr Down:\n");
            for (int ii=0; ii < *cols; ii++)
            {
                printf("%3.1f ",nu[ii]);
            }
            printf("\n");
            for (int ii=0; ii < *cols; ii++)
            {
                printf("%3c",'-');
            }
            printf("\n");
            for (int ii=0; ii < *cols; ii++)
            {
                printf("%3.1f ",nd[ii]);
            }
            printf("\n\n");
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    }
    else if (down >= 0)
    {
        MPI_Sendrecv(ld, *cols, MPI_DOUBLE, down, 0, nu, *cols, MPI_DOUBLE, down, 0, cart_comm, &status);
        newM = add_row(locM, nu, 1, rows, *cols); 
    for (int proc=0; proc<size; proc++) {
        if (proc == rank) {
            printf("Rank = %d, UD = %d, %d\n", rank, up, down);
            printf("Local Matrix:\n");
            for (int ii=0; ii<*rows; ii++) {
                for (int jj=0; jj<*cols; jj++) {
                    printf("%3.1f ",newM[ii*(*cols)+jj]);
                }
                printf("\n");
            }
            printf("\n");

            printf("Nbr Up | Local Down:\n");
            for (int ii=0; ii < *cols; ii++)
            {
                printf("%3.1f ",nu[ii]);
            }
            printf("\n");
            for (int ii=0; ii < *cols; ii++)
            {
                printf("%3c",'-');
            }
            printf("\n");
            for (int ii=0; ii < *cols; ii++)
            {
                printf("%3.1f ",ld[ii]);
            }
            printf("\n\n");
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    }
    else if (up >= 0)
    {
        MPI_Sendrecv(lu, *cols, MPI_DOUBLE, up, 0, nd, *cols, MPI_DOUBLE, up, 0, cart_comm, &status);
        newM = add_row(locM, nd, 0, rows, *cols); 
    for (int proc=0; proc<size; proc++) {
        if (proc == rank) {
            printf("Rank = %d, UD = %d, %d\n", rank, up, down);
            printf("Local Matrix:\n");
            for (int ii=0; ii<*rows; ii++) {
                for (int jj=0; jj<*cols; jj++) {
                    printf("%3.1f ",newM[ii*(*cols)+jj]);
                }
                printf("\n");
            }
            printf("\n");

            printf("Local Up | Nbr Down:\n");
            for (int ii=0; ii < *cols; ii++)
            {
                printf("%3.1f ",lu[ii]);
            }
            printf("\n");
            for (int ii=0; ii < *cols; ii++)
            {
                printf("%3c",'-');
            }
            printf("\n");
            for (int ii=0; ii < *cols; ii++)
            {
                printf("%3.1f ", nd[ii]);
            }
            printf("\n\n");
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    }

    free(nu);
    free(nd);
    free(lu);
    free(ld);
    return newM;
}

double* decompose_domain(double* M, int nr, int nc, MPI_Comm cart_comm, int lrank, int rrank, int drank, int urank)
{
    double* locM;
    int rank, size;    
    MPI_Datatype blockvec,blockvector2;
    int gridsz[2], period[2], coords[2];

    MPI_Comm_size(cart_comm, &size);
    MPI_Comm_rank(cart_comm, &rank);

    MPI_Cart_get(cart_comm, 2, gridsz, period, coords);

    int blockrows = nr / gridsz[0];
    int blockcols = nc / gridsz[1];
    locM = (double *)calloc(blockrows * blockcols, sizeof(double));

    MPI_Type_vector(blockrows, blockcols, nc, MPI_DOUBLE, &blockvector2);
    MPI_Type_create_resized( blockvector2, 0, sizeof(double), &blockvec);
    MPI_Type_commit(&blockvec);

    int displs[size];
    int counts[size];
    for (int i=0; i<gridsz[0]; i++) {
        for (int j=0; j<gridsz[1]; j++) {
            displs[i*gridsz[1]+j] = i*nc*blockrows+j*blockcols;
            counts [i*gridsz[1]+j] = 1;
        }
    }

    MPI_Scatterv(M, counts, displs, blockvec, locM, blockrows*blockcols, MPI_DOUBLE, 0, cart_comm);

    if (rrank >= 0 || lrank >= 0)
    {
        locM = comm_columns(locM, &blockrows, &blockcols, cart_comm, lrank, rrank);
    }
    MPI_Barrier(cart_comm);
    debug("%d: HERE\n", rank);
    if (drank >= 0 || urank >= 0)
    {
        locM = comm_rows(locM, &blockrows, &blockcols, cart_comm, drank, urank);
    }

    // double* padlocM = (double *)calloc((blockrows+2)*(blockcols+2), sizeof(double)); 
    // for (int i = 0; i < blockrows; i++)
    // {
    //     for (int j = 0; j < blockcols; j++)
    //     {
    //         padlocM[(i+1) * (blockcols + 2) + (j+1)] = locM[i * blockcols + j];
    //     }
    // }
    
            if (rank == 0) {
                printf("Global matrix: \n");
                for (int ii=0; ii<nr; ii++) {
                    for (int jj=0; jj<nc; jj++) {
                        printf("%3.1f ",M[ii*nc+jj]);
                    }
                    printf("\n");
                }
            printf("\n");
            }

    return locM;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm_2d;
    int ndim = 2;
    int size;
    int periodic[ndim];
    int cart_rank;
    int dimensions[ndim];
    int grid_coords[ndim];
    int rank_up, rank_down, rank_left, rank_right;

    dimensions[0] = dimensions[1] = 0;
    periodic[0] = periodic[1] = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Dims_create(size, ndim, dimensions);
    MPI_Cart_create(MPI_COMM_WORLD, ndim, dimensions, periodic, 1, &comm_2d);
    MPI_Comm_rank(comm_2d, &cart_rank);
    MPI_Cart_coords(comm_2d, cart_rank, ndim, grid_coords);

    MPI_Cart_shift(comm_2d, 1, +1, &cart_rank, &rank_right);
    MPI_Cart_shift(comm_2d, 1, -1, &cart_rank, &rank_left);
    MPI_Cart_shift(comm_2d, 0, +1, &cart_rank, &rank_down);
    MPI_Cart_shift(comm_2d, 0, -1, &cart_rank, &rank_up);

    int nr, nc, iter;
    double eb, *error_h, *error_d, *toterr_d;
    double *M, *M_d, *U_d, *M_h, *U_h, *cu_out;
    nr = atoi(argv[1]);
    nc = atoi(argv[2]);
    iter = atoi(argv[3]);
    eb = atof(argv[4]);
    //printf("%f\n", eb);

    M_h = create_matrix<double>(nr, nc, 5);
    M = create_matrix<double>(nr, nc, 5);
    U_h     = (double *)calloc(nr * nc, sizeof(double));
    cu_out  = (double *)calloc(nr * nc, sizeof(double));
    error_h = (double *)calloc(nr * nc, sizeof(double));

    //init cuda memory
    checkCudaErrors(cudaMalloc(&M_d, nr * nc * sizeof(double)));
    checkCudaErrors(cudaMalloc(&U_d, nr * nc * sizeof(double)));
    checkCudaErrors(cudaMalloc(&error_d, nr * nc * sizeof(double)));
    checkCudaErrors(cudaMalloc(&toterr_d, 24 * sizeof(double)));

    checkCudaErrors(cudaMemcpy(M_d, M_h, sizeof(double) * nr * nc, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(U_d, U_h, sizeof(double) * nr * nc, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(error_d, error_h, sizeof(double) * nr * nc, cudaMemcpyHostToDevice));

    // cpu version
    if (cart_rank == 0)
    {
        std::cout << "STARTING SERIAL..." <<std::endl;
        auto start = high_resolution_clock::now();
        cpu_jacobi<double>(M_h, eb, &iter, nr, nc);
        auto stop = high_resolution_clock::now();

        auto cpu_duration = duration_cast<duration<double>>(stop - start);

        // cuda version
    
        std::cout << "STARTING CUDA..." <<std::endl;
        start = high_resolution_clock::now();
        launch_jacobi(M_d, U_d, error_d, toterr_d, eb, iter, nr, nc);
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
    }

    double* locM = decompose_domain(M, nr, nc, comm_2d, rank_left, rank_right, rank_down, rank_up);


    free(M);
    free(M_h);
    free(U_h);
    free(cu_out);
    checkCudaErrors(cudaFree(M_d));
    checkCudaErrors(cudaFree(U_d));
    checkCudaErrors(cudaFree(error_d));
    MPI_Finalize();
    return 0;
}