/*
 * CudaBuffer.cpp
 *
 *  Created on: Aug 6, 2014
 *      Author: Sheng Lundquist
 */

#include "CudaBuffer.hpp"
#include "cuda_util.hpp"
#include <cmath>
#include <ctime>
#include <sys/time.h>

namespace PVCuda {

CudaBuffer::CudaBuffer(size_t inSize, struct cudaDeviceProp *deviceProps, cudaStream_t stream) {
   handleError(cudaMalloc(&d_ptr, inSize), "CudaBuffer constructor");
   if (!d_ptr) {
      Fatal().printf("Cuda Buffer allocation error\n");
   }
   mSize        = inSize;
   mDeviceProps = deviceProps;
   mStream      = stream;
}

CudaBuffer::~CudaBuffer() { handleError(cudaFree(d_ptr), "Freeing device pointer"); }

int CudaBuffer::copyToDevice(const void *h_ptr) {
   handleError(
         cudaMemcpyAsync(d_ptr, h_ptr, mSize, cudaMemcpyHostToDevice, mStream),
         "Copying buffer to device");
   return 0;
}

int CudaBuffer::copyToDevice(const void *h_ptr, size_t in_size, size_t offset) {
   FatalIf(
         in_size + offset > mSize,
         "copyToDevice, in_size + offset of %zu is bigger than buffer size of %zu.\n",
         in_size + offset,
         mSize);
   void *d_ptr_offset = (void *)&((char *)d_ptr)[offset];
   handleError(
         cudaMemcpyAsync(d_ptr_offset, h_ptr, mSize, cudaMemcpyHostToDevice, mStream),
         "Copying buffer with offset to device");
   return 0;
}

int CudaBuffer::copyFromDevice(void *h_ptr) {
   copyFromDevice(h_ptr, mSize);
   return 0;
}

int CudaBuffer::copyFromDevice(void *h_ptr, size_t in_size) {
   if (in_size > mSize) {
      Fatal().printf(
            "copyFromDevice: in_size of %zu is bigger than buffer size of %zu\n", in_size, mSize);
   }
   handleError(
         cudaMemcpyAsync(h_ptr, d_ptr, in_size, cudaMemcpyDeviceToHost, mStream),
         "Copying buffer from device");
   return 0;
}

void CudaBuffer::permuteWeightsPVToCudnn(
      void *d_inPtr,
      int numArbors,
      int numKernels,
      int nxp,
      int nyp,
      int nfp) {
   // outFeatures is number of kernels
   int outFeatures = numKernels;

   // Rest is patch sizes
   int ny         = nyp;
   int nx         = nxp;
   int inFeatures = nfp;

   // Calculate grid and work size
   int numWeights = numArbors * outFeatures * ny * nx * inFeatures;
   int blockSize  = mDeviceProps->maxThreadsPerBlock;
   // Ceil to get all weights
   int gridSize = std::ceil((float)numWeights / blockSize);
   // Call function
   callCudaPermuteWeightsPVToCudnn(
         gridSize, blockSize, d_inPtr, numArbors, outFeatures, ny, nx, inFeatures);
   handleCallError("Permute weights PV to CUDNN");
}

} // namespace PV
