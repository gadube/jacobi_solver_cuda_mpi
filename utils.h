#ifndef UTILS_H
#define UTILS_H
#include <cuda.h> // your system must have nvcc.
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>
#include <cstdio> 
#include <iostream>

#define checkCudaErrors(ans) check( (ans), #ans, __FILE__, __LINE__)

#define NOT_DEBUG
#ifdef DEBUG
#define debug(fmt, ...) fprintf(stderr, fmt, __VA_ARGS__)
#else
#define debug(fmt, ...)
#endif

template<typename T>
void check(T err, const char* const func, const char* const file, const int line){
    if(err != cudaSuccess){
        std::cerr << "CUDA error at:: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

#endif 