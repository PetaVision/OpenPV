/*
 * CudaBuffer.cpp
 *
 *  Created on: Aug 6, 2014
 *      Author: Sheng Lundquist
 */

#include "CudaBuffer.hpp"
#include "CudaDevice.hpp"
#include "cuda_util.hpp"
#include <sys/time.h>
#include <ctime>
#include <cmath>

namespace PVCuda {

CudaBuffer::CudaBuffer(size_t inSize, CudaDevice * inDevice, cudaStream_t stream)
{
   handleError(cudaMalloc(&d_ptr, inSize), "CudaBuffer constructor");
   if(!d_ptr){
      pvError().printf("Cuda Buffer allocation error\n");
   }
   this->size = inSize;
   this->stream = stream;
   this->device = inDevice;
}

CudaBuffer::CudaBuffer(){
   d_ptr = nullptr;
   size = 0;
   stream = nullptr;
   device = nullptr;
}

CudaBuffer::~CudaBuffer()
{
   handleError(cudaFree(d_ptr), "Freeing device pointer");
}
   
int CudaBuffer::copyToDevice(const void * h_ptr)
{
   copyToDevice(h_ptr, this->size);
   return 0;
}

int CudaBuffer::copyToDevice(const void * h_ptr, size_t in_size)
{
   if(in_size > this->size){
      pvError().printf("copyToDevice, in_size of %zu is bigger than buffer size of %zu\n", in_size, this->size);
   }
   handleError(cudaMemcpyAsync(d_ptr, h_ptr, in_size, cudaMemcpyHostToDevice, stream), "Copying buffer to device");
   return 0;
}

int CudaBuffer::copyFromDevice(void * h_ptr)
{
   copyFromDevice(h_ptr, this->size);
   return 0;
}

int CudaBuffer::copyFromDevice(void * h_ptr, size_t in_size)
{
   if(in_size > this->size){
      pvError().printf("copyFromDevice: in_size of %zu is bigger than buffer size of %zu\n", in_size, this->size);
   }
   handleError(cudaMemcpyAsync(h_ptr, d_ptr, in_size, cudaMemcpyDeviceToHost, stream), "Copying buffer from device");
   return 0;
}

void CudaBuffer::permuteWeightsPVToCudnn(void * d_inPtr, int numArbors, int numKernels, int nxp, int nyp, int nfp){
   //outFeatures is number of kernels
   int outFeatures = numKernels;

   //Rest is patch sizes
   int ny = nyp;
   int nx = nxp;
   int inFeatures = nfp;

   //Calculate grid and work size
   int numWeights = numArbors * outFeatures * ny * nx * inFeatures;
   int blockSize = device->get_max_threads();
   //Ceil to get all weights
   int gridSize = std::ceil((float)numWeights/blockSize);
   //Call function
   callCudaPermuteWeightsPVToCudnn(gridSize, blockSize, d_inPtr, numArbors, outFeatures, ny, nx, inFeatures);
   handleCallError("Permute weights PV to CUDNN");
}

} // namespace PV
