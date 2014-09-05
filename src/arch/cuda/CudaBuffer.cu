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
   handleError(cudaMalloc(&d_ptr, inSize));
   this->size = inSize;
   this->stream = stream;
}

CudaBuffer::CudaBuffer(){
   d_ptr = NULL;
   size = 0;
}

CudaBuffer::~CudaBuffer()
{
   handleError(cudaFree(d_ptr));
}
   
int CudaBuffer::copyToDevice(void * h_ptr)
{
   //handleError(cudaMemcpyAsync(d_ptr, h_ptr, size, cudaMemcpyHostToDevice, stream));
   handleError(cudaMemcpyAsync(d_ptr, h_ptr, size, cudaMemcpyHostToDevice, stream));
   return 0;
}

int CudaBuffer::copyFromDevice(void * h_ptr)
{
   handleError(cudaMemcpyAsync(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost, stream));
   return 0;
}

} // namespace PV
