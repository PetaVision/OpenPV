#ifndef _CUDAWRAPPER_HPP_
#define _CUDAWRAPPER_HPP_

#include "arch/cuda/CudaMatrix.hpp"

template <typename T>
struct CudaWrapper {
  DenseMatrix<T> dense;
  SparseMatrix<T> sparse;
  CudaWrapper() {}
  CudaWrapper(const MatrixInfo& params) { dense.init(params); }
  void dense2sparse() { sparse.fromDense(dense.getDeviceData()); }
  void sparse2dense() { sparse.toDense(dense.getDeviceData()); }
};

#endif
