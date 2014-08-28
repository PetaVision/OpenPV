/*
 * CudaKernel.hpp
 *
 *  Created on: Aug 5, 2014
 *      Author: Sheng Lundquist
 */

#ifndef CUDAKERNEL_HPP_
#define CUDAKERNEL_HPP_

#include "CudaDevice.hpp"
#include <stdlib.h>

namespace PVCuda {

class CudaKernel {
public:
   CudaKernel(CudaDevice* inDevice);
   CudaKernel();
   virtual ~CudaKernel();


   //These wrapper functions set the global/local work size member variables
   //1 dimension runs
   int run(int global_work_size); //Default local work size of 1
   int run(int global_work_size, int local_work_size);
   //2 dimension runs
   int run(int gWorkSizeX, int gWorkSizeY, int lWorkSizeX, int lWorkSizeY);
   //3 dimension runs
   int run(int gWorkSizeX, int gWorkSizeY, int gWorkSizeF,
           int lWorkSizeX, int lWorkSizeY, int lWorkSizeF);
protected:
   //This is the function that should be overwritten in child classes
   virtual int run() = 0;
   dim3 grid_size;
   dim3 block_size;
   CudaDevice* device;
   void setArgsFlag(){argsSet = true;}

private:
   void setDims(int gWorkSizeX, int gWorkSizeY, int gWorkSizeF, int lWorkSizeX, int lWorkSizeY, int lWorkSizeF);

   //argsSet must be set to true before being called
   bool argsSet;
   bool dimsSet;
};

}

#endif /* CLKERNEL_HPP_ */
