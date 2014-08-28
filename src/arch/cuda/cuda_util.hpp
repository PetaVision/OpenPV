#ifndef CUDA_UTIL_HPP_
#define CUDA_UTIL_HPP_

namespace PVCuda{
#include <stdio.h>

inline void handleError(cudaError_t error){
   if(error == cudaSuccess){
      return;
   }
   else{
      printf("%s\n", cudaGetErrorString(error));
      exit(-1);
   }
}

inline void handleCallError(){
   handleError(cudaGetLastError());
}

}

#endif
