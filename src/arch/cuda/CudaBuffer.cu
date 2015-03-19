/*
 * CudaBuffer.cpp
 *
 *  Created on: Aug 6, 2014
 *      Author: Sheng Lundquist
 */

#include "CudaBuffer.hpp"
#include "cuda_util.hpp"
#include <sys/time.h>
#include <ctime>

namespace PVCuda {

CudaBuffer::CudaBuffer(size_t inSize, cudaStream_t stream)
{
   handleError(cudaMalloc(&d_ptr, inSize), "CudaBuffer constructor");
   this->size = inSize;
   this->stream = stream;
}

CudaBuffer::CudaBuffer(){
   d_ptr = NULL;
   size = 0;
}

CudaBuffer::~CudaBuffer()
{
   handleError(cudaFree(d_ptr), "Freeing device pointer");
}
   
int CudaBuffer::copyToDevice(void * h_ptr)
{
   copyToDevice(h_ptr, this->size);
   return 0;
}

int CudaBuffer::copyToDevice(void * h_ptr, size_t in_size)
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


} // namespace PV
