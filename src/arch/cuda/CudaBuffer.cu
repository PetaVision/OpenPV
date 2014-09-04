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
   
///**
// * Convert to milliseconds
// */
//long get_cpu_time() {
//   struct timeval tim;
//   //   struct rusage ru;
//   //   getrusage(RUSAGE_SELF, &ru);
//   //   tim = ru.ru_utime;
//   gettimeofday(&tim, NULL);
//   //printf("get_cpu_time: sec==%d usec==%d\n", tim.tv_sec, tim.tv_usec);
//   return ((long) tim.tv_sec)*1000000 + (long) tim.tv_usec;
//}

int CudaBuffer::copyFromDevice(void * h_ptr)
{
   //cudaEvent_t eStart, eStop;
   //float gpuTime;
   //cudaEventCreate(&eStart);
   //cudaEventCreate(&eStop);

   //cudaDeviceSynchronize();

   //cudaEventRecord(eStart, stream);
   //long start = get_cpu_time();
   handleError(cudaMemcpyAsync(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost, stream));
   //cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
   //long stop = get_cpu_time();
   //cudaEventRecord(eStop, stream);
   //cudaEventSynchronize(eStop);
   //cudaEventElapsedTime(&gpuTime, eStart, eStop);
   //printf("cpu run time: %f\n", (double)(stop-start)/1000);
   //printf("gpu run time: %f\n", gpuTime);
   return 0;
}

} // namespace PV
