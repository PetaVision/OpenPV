/*
 * CLDevice.hpp
 *
 *  Created on: Oct 24, 2009
 *      Author: Craig Rasmussen
 */

#ifndef CLDEVICE_HPP_
#define CLDEVICE_HPP_

#include "pv_opencl.h"
#include "CLBuffer.hpp"

#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////

namespace PV {
   
void print_error_code(int code);

class CLDevice {

protected:
   int device;                           // device id (normally 0 for GPU, 1 for CPU)

public:
   CLDevice(int device);

   int initialize(int device);

   CLBuffer * createBuffer(cl_mem_flags flags, size_t size, void * host_ptr);

   CLBuffer * createReadBuffer(size_t size)   { return createBuffer(CL_MEM_READ_ONLY, size, NULL); }
   CLBuffer * createWriteBuffer(size_t size)  { return createBuffer(CL_MEM_WRITE_ONLY, size, NULL); }
   CLBuffer * createBuffer(size_t size, void * host_ptr)
                                              { return createBuffer(CL_MEM_USE_HOST_PTR, size, host_ptr); }
   
#ifdef PV_USE_OPENCL

   int createKernel(const char * filename, const char * name);

   int addKernelArg(int argid, int arg);
   int addKernelArg(int argid, CLBuffer * buf);
   int addLocalArg(int argid, size_t size);
   
   int run(size_t global_work_size);
   int run(size_t gWorkSizeX, size_t gWorkSizeY, size_t lWorkSizeX, size_t lWorkSizeY);
	
//   int copyResultsBuffer(cl_mem output, void * results, size_t size);

   int query_device_info();
	
   // execution time in microseconds
   int get_execution_time()  { return elapsed; }

protected:

   int query_device_info(int id, cl_device_id device);

   cl_uint num_devices;                  // number of computing devices

   cl_device_id device_ids[MAX_DEVICES]; // compute device id
   cl_context context;                   // compute context
   cl_command_queue commands;            // compute command queue
   cl_program program;                   // compute program
   cl_kernel kernel;                     // compute kernel
   cl_event event;                       // event identifying the kernel execution instance

   size_t global;                        // global domain size for our calculation
   size_t local;                         // local domain size for our calculation

   bool profiling;                       // flag to enable profiling
   unsigned int elapsed;                 // elapsed time in microseconds

#endif /* PV_USE_OPENCL */
};

} // namespace PV

#endif /* CLDEVICE_HPP_ */
