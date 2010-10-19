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
            const char * filename, const char * name);

   int setKernelArg(int argid, int arg);
   int setKernelArg(int argid, float arg);
   int setKernelArg(int argid, CLBuffer * buf);
   int setLocalArg(int argid, size_t size);

   int run(size_t global_work_size);
   int run(size_t gWorkSizeX, size_t gWorkSizeY, size_t lWorkSizeX, size_t lWorkSizeY);

   // execution time in microseconds
   int get_execution_time()  { return elapsed; }

protected:
   cl_kernel kernel;                     // compute kernel
   cl_command_queue commands;            // default command queue
   cl_device_id device;                  // device we are using
   cl_program program;                   // compute program
   cl_event event;                       // event identifying the kernel execution instance

   bool profiling;                       // flag to enable profiling
   unsigned int elapsed;                 // elapsed time in microseconds
};

}

#endif /* CLKERNEL_HPP_ */
