/*
 * CLDevice.hpp
 *
 *  Created on: Oct 24, 2009
 *      Author: Craig Rasmussen
 */

#ifndef CLDEVICE_HPP_
#define CLDEVICE_HPP_

#include "../../include/pv_arch.h"

#ifdef PV_USE_OPENCL

#include "CLBuffer.hpp"

#include <OpenCL/opencl.h>

////////////////////////////////////////////////////////////////////////////////

// guess at maxinum number of devices (CPU + GPU)
//
#define MAX_DEVICES (2)

// guess at maximum work item dimensions
//
#define MAX_WORK_ITEM_DIMENSIONS (3)

#define PVCL_GET_DEVICE_ID_FAILURE    1
#define PVCL_CREATE_CONTEXT_FAILURE   2
#define PVCL_CREATE_CMD_QUEUE_FAILURE 3
#define PVCL_CREATE_PROGRAM_FAILURE   4
#define PVCL_BUILD_PROGRAM_FAILURE    5
#define PVCL_CREATE_KERNEL_FAILURE    6


////////////////////////////////////////////////////////////////////////////////

namespace PV {
   
void print_error_code(int code);

class CLDevice {
public:
   CLDevice(int device);
   virtual ~CLDevice();

   int initialize(int device);

   int createKernel(const char * filename, const char * name);

   CLBuffer * createBuffer(cl_mem_flags flags, size_t size, void * host_ptr);
   
   CLBuffer * createReadBuffer(size_t size)   { return createBuffer(CL_MEM_READ_ONLY, size, NULL); }
   CLBuffer * createWriteBuffer(size_t size)  { return createBuffer(CL_MEM_WRITE_ONLY, size, NULL); }
   CLBuffer * createBuffer(size_t size, void * host_ptr)
                                              { return createBuffer(CL_MEM_USE_HOST_PTR, size, host_ptr); }
   
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

   int device;                           // device index
   
   bool profiling;                       // flag to enable profiling
   unsigned int elapsed;                 // elapsed time in microseconds
};

} // namespace PV

#endif /* PV_USE_OPENCL */
#endif /* CLDEVICE_HPP_ */
