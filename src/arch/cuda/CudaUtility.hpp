#ifndef _CUDAUTILITY_HPP_
#define _CUDAUTILITY_HPP_
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>

namespace GPULCA {

template <typename T>
void CudaMatrixNegate(int size, T* input, T* output);
}

/*  y = alpha * x + beta * y */
template <typename T>
void CudaMatrixAdd(int size, T alpha, T* x, T beta, T* y);


void CudaSparseMatrixInd2VectorInd(int nnz, int numCols, int* cooRowInd,
                                   int* cooColInd, int* output);

template <typename T>
void activationFunc(T* u, T* a, int n, T VThresh, T Amin, T Amax, T AShift,
                    T VTW, T tanTheta);

template <typename T>
void permuteWeight(int size, T* w, T* wt, int* map);

#endif
