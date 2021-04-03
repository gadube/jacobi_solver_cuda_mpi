#ifndef JACOBI_H_
#define JACOBI_H_

#include "utils.h"
#include <mpi.h>

void launch_jacobi(double* M_d, double* U_d, double* error_d, double* toterr, double eb, int maxIter, int numRows, int numCols);


#endif