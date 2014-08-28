/*
 * CudaBuffer.cpp
 *
 *  Created on: Aug 6, 2014
 *      Author: Sheng Lundquist
 */

#include "CudaBuffer.hpp"
#include "cuda_util.hpp"

namespace PVCuda {

CudaBuffer::CudaBuffer(size_t inSize)
{
   handleError(cudaMalloc(&d_ptr, inSize));
   this->size = inSize;
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
   handleError(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
   return 0;
}
   
int CudaBuffer::copyFromDevice(void * h_ptr)
{
   handleError(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost));
   return 0;
}

} // namespace PV
