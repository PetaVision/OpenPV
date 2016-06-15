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

/**
 * The base class of any kernels written in Cuda
 */
class CudaKernel {
public:
   /**
    * A constructor for the cuda kernel
    * @param inDevice The CudaDevice object this kernel will run on
    */
   CudaKernel(CudaDevice* inDevice);
   CudaKernel();
   virtual ~CudaKernel();

   /**
    * Wrapper function to run the kernel where the kernel code takes care of the size of the gpu run
    * Note that dimsSet is false if this is called
    */
   int run();

   /**
    * Wrapper function to run the kernel with the specified 1 dimensional global work size with no local work groups
    * @param global_work_size The global work size of the problem
    */
   int run(long global_work_size); //Default local work size of 1

   /**
    * Wrapper function to run the kernel with the specified 1 dimensional global work size with the specified local work groups
    * @param global_work_size The global work size of the problem
    * @param local_work_size The local work size of the problem
    */
   int run(long global_work_size, long local_work_size);
   int run_nocheck(long global_work_size, long local_work_size);

   /**
    * Wrapper function to run the kernel with the specified 2 dimensional global work size with the specified local work groups
    * @param gWorkSizeX The global work size in the X dimension
    * @param gWorkSizeY The global work size in the Y dimension
    * @param lWorkSizeX The local work size in the X dimension
    * @param lWorkSizeY The local work size in the Y dimension
    */
   int run(long gWorkSizeX, long gWorkSizeY, long lWorkSizeX, long lWorkSizeY);
   int run_nocheck(long gWorkSizeX, long gWorkSizeY, long lWorkSizeX, long lWorkSizeY);

   /**
    * Wrapper function to run the kernel with the specified 3 dimensional global work size with the specified local work groups
    * @param gWorkSizeX The global work size in the X dimension
    * @param gWorkSizeY The global work size in the Y dimension
    * @param gWorkSizeF The global work size in the F dimension
    * @param lWorkSizeX The local work size in the X dimension
    * @param lWorkSizeY The local work size in the Y dimension
    * @param lWorkSizeF The local work size in the F dimension
    */
   int run(long gWorkSizeX, long gWorkSizeY, long gWorkSizeF,
           long lWorkSizeX, long lWorkSizeY, long lWorkSizeF);
protected:
   /**
    * This virtual function should be overwritten by any subclasses. Note that argsSet and dimsSet should be set before this function is called
    */
   virtual int do_run() = 0;

   /**
    * A flag setter to tell CudaKernel that the arguments to the kernel is set
    */
   void setArgsFlag(){argsSet = true;}

   dim3 grid_size;
   dim3 block_size;
   CudaDevice* device;

   //argsSet must be set to true before being called
   bool argsSet;
   bool dimsSet;
   char const * kernelName;

private:
   /**
    * A function to set the dimensions of the run
    */
   void setDims(long gWorkSizeX, long gWorkSizeY, long gWorkSizeF, long lWorkSizeX, long lWorkSizeY, long lWorkSizeF, bool error = true);

};

}

#endif /* CLKERNEL_HPP_ */
