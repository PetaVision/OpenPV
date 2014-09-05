#ifndef CUDA_UTIL_HPP_
#define CUDA_UTIL_HPP_

namespace PVCuda{
#include <stdio.h>

/**
 * A function to handle errors from cuda api calls
 */
inline void handleError(cudaError_t error){
   if(error == cudaSuccess){
      return;
   }
   else{
      printf("%s\n", cudaGetErrorString(error));
      exit(-1);
   }
}

/**
 * A function to handle errors from cuda api calls that doesn't return an error code
 * (such as launching a kernel)
 */
inline void handleCallError(){
   handleError(cudaGetLastError());
}

}

#endif
