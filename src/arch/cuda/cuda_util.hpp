#ifndef CUDA_UTIL_HPP_
#define CUDA_UTIL_HPP_

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>

namespace PVCuda{

/**
 * A function to handle errors from cuda api calls
 */
inline void handleError(cudaError_t error, const char* message){
   if(error == cudaSuccess){
      return;
   }
   else{
      printf("Cuda call error in %s: %s\n", message, cudaGetErrorString(error));
      exit(-1);
   }
}

/**
 * A function to handle errors from cuda api calls that don't return an error code
 * (such as launching a kernel)
 */
inline void handleCallError(const char * message){
   handleError(cudaGetLastError(), message);
}

}  // end namespace PVCuda

#endif
