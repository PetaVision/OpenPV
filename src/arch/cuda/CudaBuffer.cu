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

//Weights need to be reversed for cudnn
//No need to account for many because the PV representation matches with how gsyn was reshaped.
__global__
void CudaPermuteWeightsPVToCudnn(float* dest, float* src, int numArbors, int outFeatures, int ny, int nx, int inFeatures){
   //Parameter dimensions are PV source dimensions
   int kSrc = (blockIdx.x * blockDim.x) + threadIdx.x;
   if(kSrc < outFeatures * ny * nx * inFeatures){
      int kA = kSrc/(outFeatures*ny*nx*inFeatures);
      int kOF = (kSrc % (outFeatures*ny*nx*inFeatures))/(ny*nx*inFeatures);
      int kY  = (kSrc % (ny*nx*inFeatures))/(nx*inFeatures);
      int kX  = (kSrc % (nx*inFeatures))/inFeatures;
      int kIF = (kSrc % inFeatures);

      int sA  = outFeatures * inFeatures * ny * nx;
      int sOF = inFeatures * ny * nx;
      int sIF = ny * nx;
      int sY  = nx;

      int kDest = kA * sA + kOF * sOF + kIF * sIF + (ny-kY-1) * sY + (nx-kX-1);

      dest[kDest] = src[kSrc];
   }
}

namespace PVCuda {

CudaBuffer::CudaBuffer(size_t inSize, CudaDevice * inDevice, cudaStream_t stream)
{
   handleError(cudaMalloc(&d_ptr, inSize), "CudaBuffer constructor");
   if(!d_ptr){
      printf("Cuda Buffer allocation error\n");
      exit(-1);
   }
   this->size = inSize;
   this->stream = stream;
   this->device = inDevice;
}

CudaBuffer::CudaBuffer(){
   d_ptr = NULL;
   size = 0;
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
      printf("copyToDevice, in_size of %zu is bigger than buffer size of %zu\n", in_size, this->size);
      exit(-1);
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
      printf("copyFromDevice: in_size of %zu is bigger than buffer size of %zu\n", in_size, this->size);
      exit(-1);
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
   int gridSize = ceil((float)numWeights/blockSize);
   //Call function
   //printf("Calling weights PV To Cudnn with (%d, %d, %d, %d)\n", outFeatures, ny, nx, inFeatures);
   CudaPermuteWeightsPVToCudnn<<<gridSize, blockSize, 0, stream>>>((float*)d_ptr, (float*)d_inPtr, numArbors, outFeatures, ny, nx, inFeatures);
   handleCallError("Permute weights PV to CUDNN");
}

} // namespace PV
