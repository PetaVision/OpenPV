#ifndef _CUDAVECTOR_HPP_
#define _CUDAVECTOR_HPP_
#include <algorithm>
#include <vector>
#include <exception>
#include <stdexcept>
#include <cuda_runtime.h>
#include <utils/PVLog.hpp>

using namespace std;

namespace PVCuda {

template <class T>
class CudaVector {
 public:
  CudaVector() : size(0), bufFlag(false), deviceData(nullptr) {}

  // Only initialize host
  CudaVector(int n, bool flag = false)
      : size(n), bufFlag(flag), deviceData(nullptr) {
    hostVector.resize(n);
    cudaMallocCheck((void **)&deviceData, n);
  }

  // Initialize host and device
  CudaVector(int n, const T &x, bool flag = false)
      : size(n), bufFlag(flag), deviceData(nullptr) {
    hostVector.resize(n);
    fill(hostVector.begin(), hostVector.end(), x);
    cudaMallocCheck((void **)&deviceData, n);
    host2Device();
  }

  // Initialize host and device
  CudaVector(const vector<T> &vec, bool flag = false)
      : size(vec.size()), deviceData(nullptr) {
    hostVector.resize(vec.size());
    copy(vec.begin(), vec.end(), hostVector.begin());
    cudaMallocCheck((void **)&deviceData, size);
    host2Device();
  }

  // Initialize device only, the pointer is assumed be pointing to a device
  // memory.
  CudaVector(int n, const T *vec, bool flag = false)
      : size(n), bufFlag(flag), deviceData(nullptr) {
    cudaMallocCheck((void **)&deviceData, size);
    cudaMemcpyCheck(deviceData, vec, size, cudaMemcpyDeviceToDevice);
  }

  ~CudaVector() {
    if (deviceData != nullptr && size != 0)
      cudaFreeDestructorCheck((void **)&deviceData);
  }

  void host2Device() {
    if (hostVector.empty())
      throw logic_error("Error host2Device: hostVector is empty.\n");

    if (deviceData == nullptr) cudaMallocCheck((void **)&deviceData, size);

    cudaMemcpyCheck(deviceData, hostVector.data(), size,
                    cudaMemcpyHostToDevice);
  }

  void device2Host() {
    if (deviceData == nullptr)
      throw logic_error("Error device2Host: deviceData is empty.\n");

    if (hostVector.empty()) hostVector.resize(size);

    cudaMemcpyCheck(hostVector.data(), deviceData, size,
                    cudaMemcpyDeviceToHost);
  }

  CudaVector<T> &operator=(const CudaVector<T> &obj) {
    if (this == &obj) return *this;

    if (isBuf()) {  // does not change size
      if (getSize() < obj.getSize())
        throw logic_error(
            "The size of new values is greater than buffer size.");
      copy(obj.hostVector.begin(), obj.hostVector.end(), hostVector.begin());
    } else if (size == obj.size) {
      hostVector = obj.hostVector;
    } else {
      if (deviceData != nullptr) cudaFreeCheck((void **)&deviceData);
      size = obj.size;
      hostVector = obj.hostVector;
      cudaMallocCheck((void **)&deviceData, size);
    }
    cudaMemcpyCheck(
        deviceData, obj.deviceData,
        obj.size,  // The cudamemcpy always copy obj.size of elements.
        cudaMemcpyDeviceToDevice);
    return *this;
  }

  const bool &isBuf() const { return bufFlag; }
  vector<T> &getHostVector() { return hostVector; }
  const vector<T> &getHostVector() const { return hostVector; }
  T *getHostData() { return hostVector.data(); }
  const T *getHostData() const { return hostVector.data(); }
  T *getDeviceData() { return deviceData; }
  const T *getDeviceData() const { return deviceData; }

  bool empty() {
    if (size == 0)
      return true;
    else
      return false;
  }

  const int &getSize() const { return size; }

  void resize(int n) {
    if (size == n)
      return;
    else if (size > n) {
      size = n;
      return;
    } else {
      size = n;
      hostVector.resize(n);
      if (deviceData != nullptr) cudaFreeCheck((void **)&deviceData);
      cudaMallocCheck((void **)&deviceData, n);
    }
  }

  void resize(int n, bool flag) {
    bufFlag = flag;
    if (size == n)
      return;
    else if (size > n) {
      size = n;
      return;
    } else {
      size = n;
      hostVector.resize(n);
      if (deviceData != nullptr) cudaFreeCheck((void **)&deviceData);
      cudaMallocCheck((void **)&deviceData, n);
    }
  }

  void setHostVector(const vector<T> &vec) {
    if (isBuf()) {
      if (getSize() < vec.size())
        throw logic_error(
            "The size of new values is greater than buffer size.");
      copy(vec.begin(), vec.end(), hostVector.begin());
    } else {
      hostVector = vec;
      if (size < vec.size()) {
        if (deviceData != nullptr) cudaFreeCheck((void **)&deviceData);
        cudaMallocCheck((void **)&deviceData, vec.size());
      }
      size = vec.size();
    }
  }

  void setHostVector(const CudaVector<T> &vec) {
    if (isBuf()) {
      if (getSize() < vec.size())
        throw logic_error(
            "The size of new values is greater than buffer size.");
      copy(vec.hostVector.begin(), vec.hostVector.end(), hostVector.begin());
    } else {
      hostVector = vec.getHostVec();
      if (size < vec.getSize()) {
        if (deviceData != nullptr) cudaFreeCheck((void **)&deviceData);
        cudaMallocCheck((void **)&deviceData, vec.getSize());
      }
      size = vec.getSize();
    }
  }

  void setHostVector(const int &n, const T *vec) {
    setDeviceData(n, vec);
    device2Host();
  }

  void setDeviceData(const T &data) {
    cudaMemsetCheck(deviceData, (int)data, size);
  }

  void setDeviceData(const vector<T> &vec) {
    setHostVector(vec);
    host2Device();
  }

  void setDeviceData(const int &n, const T *vec) {
    if (isBuf()) {
      if (getSize() < n)
        throw logic_error(
            "The size of new values is greater than buffer size.");
    } else {
      if (size < n) {
        if (deviceData != nullptr) cudaFreeCheck((void **)&deviceData);
        cudaMallocCheck((void **)&deviceData, n);
      }
      hostVector.resize(n);
      size = n;
    }
    cudaMemcpyCheck(deviceData, vec, n, cudaMemcpyDeviceToDevice);
  }

  void setDeviceData(const CudaVector<T> &vec) {
    if (this->isBuf()) {
      if (getSize() < vec.getSize())
        throw logic_error(
            "The size of new values is greater than buffer size.");
    } else {
      if (size < vec.getSize()) {
        if (deviceData != nullptr) cudaFreeCheck((void **)&deviceData);
        cudaMallocCheck((void **)&deviceData, vec.getSize());
      }
      hostVector.resize(vec.getSize());
      size = vec.getSize();
    }
    cudaMemcpyCheck(deviceData, vec.getDeviceData(), vec.getSize(),
                    cudaMemcpyDeviceToDevice);
  }

  void CUDAHostVectorFree() { hostVector.clear(); }

  void CUDADeviceDataFree() {
    if (deviceData != nullptr) cudaFreeCheck((void **)&deviceData);
    deviceData = nullptr;
  }

  void CudaVectorFree() {
    size = 0;
    CUDAHostVectorFree();
    CUDADeviceDataFree();
  }

 protected:
  void cudaMallocCheck(void **p, const int &size) {
    cudaError_t cudaError = cudaMalloc(p, size * sizeof(T));
    if (cudaError != cudaSuccess)
      throw runtime_error("CUDA malloc error :" +
                          string(cudaGetErrorName(cudaError)));
  }

  void cudaFreeCheck(void **p) {
    cudaError_t cudaError = cudaFree(*p);
    if (cudaError != cudaSuccess)
      throw runtime_error("CUDAFree error :" +
                          string(cudaGetErrorName(cudaError)));
    *p = nullptr;
  }

  void cudaFreeDestructorCheck(void **p) {
    cudaError_t cudaError = cudaFree(*p);
    pvError() << ("CUDAFree error :" + string(cudaGetErrorName(cudaError))) << endl;
    *p = nullptr;
  }

  void cudaMemcpyCheck(void *dst, const void *src, const int &count,
                       cudaMemcpyKind kind) {
    cudaError_t cudaError = cudaMemcpy(dst, src, count * sizeof(T), kind);
    string s;

    switch (kind) {
      case 0:
        s = "cudaMemcpyHostToHost";
        break;
      case 1:
        s = "cudaMemcpyHostToDevice";
        break;
      case 2:
        s = "cudaMemcpyDeviceToHost";
        break;
      case 3:
        s = "cudaMemcpyDeviceToDevice";
        break;
      default:
        break;
    }
    if (cudaError != cudaSuccess)
      throw runtime_error("cudaMemcpy error: " +
                          string(cudaGetErrorName(cudaError)) + " " + s);
  }

  void cudaMemsetCheck(void *devPtr, int v, int count) {
    cudaError_t cudaError = cudaMemset(devPtr, v, count * sizeof(T));
    if (cudaError != cudaSuccess)
      throw runtime_error("CUDA memset error :" +
                          string(cudaGetErrorName(cudaError)));
  }

 private:
  int size;
  bool bufFlag;
  vector<T> hostVector;
  T *deviceData;
};
}
#endif
