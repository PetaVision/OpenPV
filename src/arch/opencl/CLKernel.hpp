/*
 * CLKernel.hpp
 *
 *  Created on: Aug 1, 2010
 *      Author: Craig Rasmussen
 */

#ifndef CLKERNEL_HPP_
#define CLKERNEL_HPP_

#include "CLBuffer.hpp"
#include <stdlib.h>

namespace PV {

class CLKernel {
public:
   CLKernel(cl_context context, cl_command_queue commands, cl_device_id device,
            const char * filename, const char * name, const char * options);
   virtual ~CLKernel();

   int setKernelArg(unsigned int arg_index, size_t arg_size, const void * arg_value);

   int setKernelArg(int argid, int arg)         {return setKernelArg(argid, sizeof(int), &arg);}
   int setKernelArg(int argid, size_t arg)      {return setKernelArg(argid, sizeof(size_t), &arg);}
   int setKernelArg(int argid, float arg)       {return setKernelArg(argid, sizeof(float), &arg);}
   int setKernelArg(int argid, double arg)      {return setKernelArg(argid, sizeof(double), &arg);}
   int setKernelArg(int argid, CLBuffer * buf);
   int setLocalArg(int argid, size_t size);

   int run(size_t global_work_size,
         unsigned int nWait, cl_event * waitList, cl_event * ev);
   int run(size_t global_work_size, size_t local_work_size,
         unsigned int nWait, cl_event * waitList, cl_event * ev);
   int run(size_t gWorkSizeX, size_t gWorkSizeY, size_t lWorkSizeX, size_t lWorkSizeY,
           unsigned int nWait, cl_event * waitList, cl_event * ev);
   int run(size_t gWorkSizeX, size_t gWorkSizeY, size_t gWorkSizeF,
                  size_t lWorkSizeX, size_t lWorkSizeY, size_t lWorkSizeF,
                  unsigned int nWait, cl_event * waitList, cl_event * ev);

//   int run(size_t global_work_size, size_t local_work_size)
//      {
//         return run(global_work_size, local_work_size, 0, NULL, NULL);
//      }

//   int run(size_t gWorkSizeX, size_t gWorkSizeY, size_t lWorkSizeX, size_t lWorkSizeY)
//      {
//         return run(gWorkSizeX, gWorkSizeY, lWorkSizeX, lWorkSizeY, 0, NULL, NULL);
//      }

   // execution time in microseconds
   int get_execution_time()  { return elapsed; }

#ifdef PV_USE_OPENCL
   int finish()              { return clFinish(commands); }
#else
   int finish()              { return CL_SUCCESS; }
#endif // PV_USE_OPENCL

protected:
   cl_command_queue commands;            // default command queue
   cl_device_id device;                  // device we are using
#ifdef PV_USE_OPENCL
   cl_program program;                   // compute program
   cl_kernel kernel;                     // compute kernel
//   cl_event event;                       // event identifying the kernel execution instance
#endif // PV_USE_OPENCL
   bool profiling;                       // flag to enable profiling
   unsigned int elapsed;                 // elapsed time in microseconds
};

}

#endif /* CLKERNEL_HPP_ */
