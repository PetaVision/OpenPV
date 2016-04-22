#ifndef CUDA_UTIL_HPP_
#define CUDA_UTIL_HPP_

namespace PVCuda{
#include <stdio.h>

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
 * A function to handle errors from cuda api calls that doesn't return an error code
 * (such as launching a kernel)
 */
inline void handleCallError(const char * message){
   handleError(cudaGetLastError(), message);
}

}

#endif
