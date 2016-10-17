/*
  4D Matrix
*/

#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_
#include <vector>
#include <exception>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cudnn.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <string>
#include "CudaVector.hpp"
#include "CudaUtility.hpp"

namespace PVCuda {

inline void cudnnStatusCheck(const cudnnStatus_t &status, string msg) {
  if (status != CUDNN_STATUS_SUCCESS)
    throw std::runtime_error("cuDNN " + msg + " error :" +
                             string(cudnnGetErrorString(status)));
}

inline void cudnnStatusDestructorCheck(const cudnnStatus_t &status,
                                       string msg) {
  if (status != CUDNN_STATUS_SUCCESS)
    pvError() << ("cuDNN " + msg + " error :" +
                  string(cudnnGetErrorString(status))) << endl;
}

inline void cusparseStatusCheck(const cusparseStatus_t &status, string msg) {
  if (status != CUSPARSE_STATUS_SUCCESS)
    throw std::runtime_error("cusparse " + msg + " error :" +
                             to_string(status));
}

inline void cusparseStatusDestructorCheck(const cusparseStatus_t &status,
                                          string msg) {
  if (status != CUSPARSE_STATUS_SUCCESS)
    pvError() << ("cusparse " + msg + " error :" + to_string(status)) << endl;
}

inline void cublasStatusCheck(const cublasStatus_t &status, string msg) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cublas " + msg + " error :" + to_string(status));
}

inline void cublasStatusDestructorCheck(const cublasStatus_t &status,
                                        string msg) {
  if (status != CUBLAS_STATUS_SUCCESS)
    pvError() << ("cublas " + msg + " error :" + to_string(status)) << endl;
}

inline void cudaStatusCheck(const cudaError_t &status, string msg) {
  if (status != cudaSuccess)
    throw runtime_error("CUDA " + msg + " error :" +
                        string(cudaGetErrorName(status)));
}

inline void cudaStatusCheck(string msg) {
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw runtime_error("CUDA " + msg + " error :" +
                        string(cudaGetErrorName(status)));
}

enum DataLayoutType { NCHW, NHWC };

struct MatrixInfo {
  int n, height, width, channel;

  DataLayoutType layout;

  MatrixInfo &operator=(const MatrixInfo &info) {
    n = info.n;
    height = info.height;
    width = info.width;
    channel = info.channel;
    layout = info.layout;
    return (*this);
  }
};  // struct MatrixInfo

class Matrix {
 public:
  Matrix() {}
  Matrix(const MatrixInfo &info) : matrixInfo(info) {
    size = info.n * info.height * info.width * info.channel;
  }

  /*  init is for the initialization of member variable */
  void init(const MatrixInfo &info) {
    matrixInfo = info;
    size = info.n * info.height * info.width * info.channel;
  }

  virtual ~Matrix() {}

  /*  Matrix information  */
  const int &getSize() const { return size; }
  const int &getN() const { return matrixInfo.n; }
  const int &getH() const { return matrixInfo.height; }
  const int &getW() const { return matrixInfo.width; }
  const int &getC() const { return matrixInfo.channel; }
  const DataLayoutType &getLayout() const { return matrixInfo.layout; }
  const int &getHost2DRows() const { return host2DRows; }
  const int &getHost2DCols() const { return host2DCols; }
  const int &getDevice2DRows() const { return device2DRows; }
  const int &getDevice2DCols() const { return device2DCols; }

  /*  GPU operations  */
  virtual void host2Device() = 0;
  virtual void device2Host() = 0;

 protected:
  void setMatrix2DInfo(int hr, int hc, int dr, int dc) {
    host2DRows = hr;
    host2DCols = hc;
    device2DRows = dr;
    device2DCols = dc;
  }

 private:
  int size;
  MatrixInfo matrixInfo;
  /*  Information of dimensions when the matrix is considered as a 2D matrix,
   * i.e. either (h*w) X c or c X (h*w) */
  int host2DRows, host2DCols, device2DRows, device2DCols;
};  // class Matrix

template <class T>
class DenseMatrix : public Matrix {
 public:
  DenseMatrix() {}
  DenseMatrix(const MatrixInfo &info) : Matrix(info) {
    cudaVec.resize(getSize());
    setDenseMatrix2DInfo();
  }
  /*  initialize using host data */
  DenseMatrix(const MatrixInfo &info, const vector<T> &hostData)
      : Matrix(info) {
    cudaVec.setDeviceData(hostData);
    setDenseMatrix2DInfo();
  }

  DenseMatrix(const MatrixInfo &info, const T *hostData) : Matrix(info) {
    cudaVec.setDeviceData(hostData);
    setDenseMatrix2DInfo();
  }

  /*  init is for the initialization of member variable */
  void init(const MatrixInfo &info) {
    Matrix::init(info);
    cudaVec.resize(getSize());
    setDenseMatrix2DInfo();
  }

  void init(const MatrixInfo &info, const vector<T> &hostData) {
    Matrix::init(info);
    cudaVec.setDeviceData(hostData);
    setDenseMatrix2DInfo();
  }

  void init(const MatrixInfo &info, const T *hostData) {
    Matrix::init(info);
    cudaVec.setDeviceData(hostData);
    setDenseMatrix2DInfo();
  }

  ~DenseMatrix() {}

  void host2Device() { cudaVec.host2Device(); }
  void device2Host() { cudaVec.device2Host(); }

  T *getHostData() { return cudaVec.getHostVector().data(); }
  const T *getHostData() const { return cudaVec.getHostVector().data(); }

  vector<T> &getHostVector() { return cudaVec.getHostVector(); }
  const vector<T> &getHostVector() const { return cudaVec.getHostVector(); }

  T *getDeviceData() { return cudaVec.getDeviceData(); }
  const T *getDeviceData() const { return cudaVec.getDeviceData(); }

  CudaVector<T> &getCudaVector() { return cudaVec; }
  const CudaVector<T> &getCudaVector() const { return cudaVec; }

  void setDenseMatrixDeviceData(const T &x) { cudaVec.setDeviceData(x); }

 private:
  CudaVector<T> cudaVec;
  void setDenseMatrix2DInfo() {
    switch (getLayout()) {
      case NCHW:
        setMatrix2DInfo(getC(), getH() * getW(), getH() * getW(), getC());
        break;
      case NHWC:
        setMatrix2DInfo(getH() * getW(), getC(), getC(), getH() * getW());
        break;
    }
  }
};  // class DenseMatrix

template <class T>
class SparseMatrix : public Matrix {
 public:
  SparseMatrix() {}
  SparseMatrix(const MatrixInfo &info, cusparseHandle_t *cusparseHandle,
               cusparseMatDescr_t *cusparseMatDescr, bool bufFlag)
      : Matrix(info),
        cusparseHandle(cusparseHandle),
        cusparseMatDescr(cusparseMatDescr),
        bufFlag(bufFlag) {
    nnz = 0;
    if (isBuf()) {
      cooCsrValA.resize(getSize(), bufFlag);
      cooRowIndA.resize(getSize(), bufFlag);
      cooColIndA.resize(getSize(), bufFlag);
      csrRowIndA.resize(getSize() + 1, bufFlag);
      csrColIndA.resize(getSize(), bufFlag);
      vecInd.resize(getSize(), bufFlag);
    }
    setSparseMatrix2DInfo();
    nnzPerRowColumn.resize(getDevice2DRows());
  }

  SparseMatrix(const MatrixInfo &info, int nnz, vector<T> hostCooCsrValA,
               vector<int> hostCooRowIndA, vector<int> hostCooColIndA,
               cusparseHandle_t *cusparseHandle,
               cusparseMatDescr_t *cusparseMatDescr, bool bufFlag)
      : Matrix(info),
        cusparseHandle(cusparseHandle),
        cusparseMatDescr(cusparseMatDescr),
        bufFlag(bufFlag) {
    this->nnz = nnz;

    if (isBuf()) {
      cooCsrValA.resize(Matrix::getSize(), bufFlag);
      cooRowIndA.resize(Matrix::getSize(), bufFlag);
      cooColIndA.resize(Matrix::getSize(), bufFlag);
      csrRowIndA.resize(Matrix::getSize() + 1, bufFlag);
      csrColIndA.resize(Matrix::getSize(), bufFlag);
      vecInd.resize(getSize(), bufFlag);
    } else {
      cooCsrValA.resize(hostCooCsrValA.size());
      cooRowIndA.resize(hostCooRowIndA.size());
      cooColIndA.resize(hostCooColIndA.size());
    }

    cooCsrValA.setDeviceData(hostCooCsrValA);
    cooRowIndA.setDeviceData(cooRowIndA);
    cooColIndA.setDeviceData(cooColIndA);

    setSparseMatrix2DInfo();
    nnzPerRowColumn.resize(getDevice2DRows());
  }

  ~SparseMatrix() {}

  /*  init is for the initialization of member variable */
  void init(const MatrixInfo &info, const cusparseHandle_t *cusparseHandle,
            const cusparseMatDescr_t *cusparseMatDescr, bool bufFlag) {
    Matrix::init(info);
    this->cusparseHandle = cusparseHandle;
    this->cusparseMatDescr = cusparseMatDescr;
    this->bufFlag = bufFlag;
    nnz = 0;
    if (isBuf()) {
      cooCsrValA.resize(getSize(), bufFlag);
      cooRowIndA.resize(getSize(), bufFlag);
      cooColIndA.resize(getSize(), bufFlag);
      csrRowIndA.resize(getSize() + 1, bufFlag);
      csrColIndA.resize(getSize(), bufFlag);
      vecInd.resize(getSize(), bufFlag);
    }
    setSparseMatrix2DInfo();
    nnzPerRowColumn.resize(getDevice2DRows());
  }

  void init(const MatrixInfo &info, int nnz, vector<T> hostCooCsrValA,
            vector<int> hostCooRowIndA, vector<int> hostCooColIndA,
            const cusparseHandle_t *cusparseHandle,
            const cusparseMatDescr_t *cusparseMatDescr, bool bufFlag) {
    Matrix::init(info);
    this->cusparseHandle = cusparseHandle;
    this->cusparseMatDescr = cusparseMatDescr;
    this->bufFlag = bufFlag;
    this->nnz = nnz;

    if (isBuf()) {
      cooCsrValA.resize(Matrix::getSize(), bufFlag);
      cooRowIndA.resize(Matrix::getSize(), bufFlag);
      cooColIndA.resize(Matrix::getSize(), bufFlag);
      csrRowIndA.resize(Matrix::getSize() + 1, bufFlag);
      csrColIndA.resize(Matrix::getSize(), bufFlag);
      vecInd.resize(getSize(), bufFlag);
    } else {
      cooCsrValA.resize(hostCooCsrValA.size());
      cooRowIndA.resize(hostCooRowIndA.size());
      cooColIndA.resize(hostCooColIndA.size());
    }

    cooCsrValA.setDeviceData(hostCooCsrValA);
    cooRowIndA.setDeviceData(cooRowIndA);
    cooColIndA.setDeviceData(cooColIndA);

    setSparseMatrix2DInfo();
    nnzPerRowColumn.resize(getDevice2DRows());
  }
  const bool &isBuf() const { return bufFlag; }

  /*  Matrix access */
  T *getCooCsrValA() { return cooCsrValA.getDeviceData(); }
  int *getCooRowIndA() { return cooRowIndA.getDeviceData(); }
  int *getCooColIndA() { return cooColIndA.getDeviceData(); }
  int *getCsrRowIndA() { return csrRowIndA.getDeviceData(); }
  int *getCsrColIndA() { return csrColIndA.getDeviceData(); }
  int *getVecInd() { return vecInd.getDeviceData(); }

  const int &getNNZ() const { return nnz; }

  int *getNNZPerRowColumn() { return nnzPerRowColumn.getDeviceData(); }
  CudaVector<T> &getCooCsrValAVector() { return cooCsrValA; }
  CudaVector<int> &getCooRowIndAVector() { return cooRowIndA; }
  CudaVector<int> &getCooColIndAVector() { return cooColIndA; }
  CudaVector<int> &getCsrRowIndAVector() { return csrRowIndA; }
  CudaVector<int> &getCsrColIndAVector() { return csrColIndA; }
  CudaVector<int> &getnnzPerRowColumnVector() { return nnzPerRowColumn; }
  CudaVector<int> &getnnzVecIndVector() { return vecInd; }

  const cusparseHandle_t *getcusparseHandle() const { return cusparseHandle; }

  /*  from host COO data to device data only */
  void host2Device() {
    /*  copy COO data to GPU */
    cooCsrValA.host2Device();
    cooRowIndA.host2Device();
    cooColIndA.host2Device();
    Coo2Csr();
  }
  /*  from devic COO data to host data only */
  void device2Host() {
    Csr2Coo();
    cooCsrValA.device2Host();
    cooRowIndA.device2Host();
    cooColIndA.device2Host();
  }

  void fromDense(T *data) {
    cusparseStatusCheck(
        cusparseSnnz(*cusparseHandle, CUSPARSE_DIRECTION_ROW, getDevice2DRows(),
                     getDevice2DCols(), *cusparseMatDescr, data,
                     getDevice2DRows(), nnzPerRowColumn.getDeviceData(), &nnz),
        "compute nnz");

    cudaStatusCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    if (!isBuf()) {
      cooCsrValA.resize(nnz);
      csrRowIndA.resize(getDevice2DRows() + 1);
      csrColIndA.resize(nnz);
    }

    cusparseStatusCheck(
        cusparseSdense2csr(
            *cusparseHandle, getDevice2DRows(), getDevice2DCols(),
            *cusparseMatDescr, data, getDevice2DRows(),
            nnzPerRowColumn.getDeviceData(), cooCsrValA.getDeviceData(),
            csrRowIndA.getDeviceData(), csrColIndA.getDeviceData()),
        "dense2csr");

    cudaStatusCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  }

  void toDense(T *data) {
    cusparseStatus_t cusparseStatus = cusparseScsr2dense(
        *cusparseHandle, getDevice2DRows(), getDevice2DCols(),
        *cusparseMatDescr, cooCsrValA.getDeviceData(), 
        csrRowIndA.getDeviceData(), csrColIndA.getDeviceData(), data, getDevice2DRows());
		cusparseStatusCheck(cusparseStatus, "csr2dense");
		cudaStatusCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
		
  }

  void sparseMatrix2Vector() {
    Csr2Coo();
    CudaSparseMatrixInd2VectorInd(
        nnz, getDevice2DRows(), cooRowIndA.getDeviceData(),
        cooColIndA.getDeviceData(), vecInd.getDeviceData());
    cudaStatusCheck("computing sparse vector index");
  }

 private:
  /*  cuSparse handler */
  const cusparseHandle_t *cusparseHandle;
  const cusparseMatDescr_t *cusparseMatDescr;
  bool bufFlag;
  int nnz;

  /*  device params  */
  CudaVector<int> nnzPerRowColumn;

  /*  COO params  */
  CudaVector<T> cooCsrValA;
  CudaVector<int> cooRowIndA, cooColIndA;

  /*  CSR params */
  CudaVector<int> csrRowIndA, csrColIndA;

  CudaVector<int> vecInd;

  /*  Inner-format transform function */
  void Coo2Csr() {
    if (cooCsrValA.empty() || cooRowIndA.empty() || cooColIndA.empty())
      throw std::logic_error("Coo2Csr, empty coo device pointers.");
    /*  allocate CSR GPU memory */
    if (!isBuf()) {
      csrRowIndA.resize(getDevice2DRows() + 1);
      csrColIndA.setDeviceData(cooColIndA);
    }

    /*  convert  */
    cusparseStatus_t cuSparseError = cusparseXcoo2csr(
        (*cusparseHandle), cooRowIndA.getDeviceData(), nnz, getDevice2DRows(),
        csrRowIndA.getDeviceData(), CUSPARSE_INDEX_BASE_ZERO);
    cusparseStatusCheck(cuSparseError, "cusparseXcoo2csr");
    cudaStatusCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  }

  void Csr2Coo() {
    if (cooCsrValA.empty() || csrRowIndA.empty() || csrColIndA.empty())
      throw std::logic_error("Csr2Coo, empty csr device pointers.");

    /*  allocate COO GPU memory */
    cooRowIndA.resize(nnz);

    /*  convert  */
    cooColIndA.setDeviceData(csrColIndA);

    cusparseStatus_t cuSparseError = cusparseXcsr2coo(
        (*cusparseHandle), csrRowIndA.getDeviceData(), nnz, getDevice2DRows(),
        cooRowIndA.getDeviceData(), CUSPARSE_INDEX_BASE_ZERO);
    cusparseStatusCheck(cuSparseError, "cusparseXcoo2csr");
    cudaStatusCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  }

  void setSparseMatrix2DInfo() {
    switch (getLayout()) {
      case NCHW:
        setMatrix2DInfo(getC(), getH() * getW(), getC(), getH() * getW());
        break;
      case NHWC:
        setMatrix2DInfo(getH() * getW(), getC(), getH() * getW(), getC());
        break;
    }
  }
};  // class SparseMatrix

}  // namespace GPUPV

#endif  // #ifndef _MATRIX_HPP_
