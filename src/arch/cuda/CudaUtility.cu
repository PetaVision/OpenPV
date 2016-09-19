#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>

template <typename T>
__global__
void CudaMatrixNegate(int size, T* input, T* output) {
  thrust::device_ptr<T> D(input), O(output);
  thrust::transform(thrust::device, D, D + size, O, thrust::negate<T>());
}

template 
__global__
void CudaMatrixNegate<float>(int size, float* input, float* output);

template <typename T>
__global__
void CudaMatrixAdd(int size, T alpha, T* x, T beta, T* y) {
  thrust::device_ptr<T> X(x), Y(y);
  auto addFunc = [=](T z1, T z2) { return z1 * alpha + z2 * beta; };
  thrust::transform(thrust::device, X, X + size, Y, Y, addFunc);
}

template 
__global__
void CudaMatrixAdd<float>(int size, float alpha, float* x, float beta, float* y);

__global__
void CudaSparseMatrixInd2VectorInd(int nnz, int numCols, int* cooRowInd,
                                   int* cooColInd, int* output) {
  thrust::device_ptr<int> R(cooRowInd), C(cooColInd), O(output);
  auto func = [=](int r, int c) { return r * numCols + c; };
  thrust::transform(thrust::device, R, R + nnz, C, O, func);
}

template <typename T>
__global__
void activationFunc(T* u, T* a, int n, T VThresh, T Amin, T Amax, T AShift,
                    T VTW, T tanTheta) {
  thrust::device_ptr<T> U(u), A(a);
  auto activationFuntor = [=](const T& u) {
    if (u < VThresh) {
      return Amin;
    } else if (u >= VThresh && u < VTW) {
      return (u - VThresh) * tanTheta;
    } else {
      T x = u - AShift;
      if (x > Amax) {
        return Amax;
      }
      return x;
    }
  };
  thrust::transform(thrust::device, U, U + n, A, activationFuntor);
}

template 
__global__
void activationFunc<float>(float* u, float* a, int n, float VThresh, float Amin, float Amax, float AShift,
                    float VTW, float tanTheta);

template <typename T>
__global__
void permuteWeight(int n, T* w, T* wt, int* map) {
  thrust::device_ptr<T> W(w), WT(wt);
  thrust::device_ptr<int> M(map);
  thrust::gather(thrust::device, M, M + n, W, WT);
}

template
__global__
void permuteWeight<float>(int n, float* w, float* wt, int* map);