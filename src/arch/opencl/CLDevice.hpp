/*
 * CLDevice.hpp
 *
 *  Created on: Oct 24, 2009
 *      Author: Craig Rasmussen
 */

#ifndef CLDEVICE_HPP_
#define CLDEVICE_HPP_

#include "pv_opencl.h"
#include <stdlib.h>

#define CL_DEVICE_DEFAULT 1

////////////////////////////////////////////////////////////////////////////////

namespace PV {
   
class CLBuffer;
class CLKernel;
class CLTimer;

class CLDevice {
   
#ifdef PV_USE_OPENCL


protected:
   int device_id;                         // device id (normally 0 for GPU, 1 for CPU)

public:
   CLDevice(int device);
   virtual ~CLDevice();

   int initialize(int device);

   static void print_error_code(int code, char * buffer, int n);

   int id()  { return device_id; }

   CLTimer* createTimer(double init_time=0.0);
   CLTimer* createTimer(const char * timermessage, double init_time=0.0);
   CLTimer* createTimer(const char * objname, const char * objtype, const char * timertype, double init_time=0.0);

   CLBuffer * createBuffer(cl_mem_flags flags, size_t size, void * host_ptr);

   CLBuffer * createReadBuffer(size_t size, void * host_ptr)
         { return createBuffer(CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, size, host_ptr); }
   CLBuffer * createWriteBuffer(size_t size, void * host_ptr)
         { return createBuffer(CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, size, host_ptr); }
   CLBuffer * createBuffer(size_t size, void * host_ptr)
         { return createBuffer(CL_MEM_COPY_HOST_PTR, size, host_ptr); }

   CLKernel * createKernel(const char * filename, const char * name, const char * options);
   CLKernel * createKernel(const char * filename, const char * name)
         { return createKernel(filename, name, NULL); }

   int syncDevice();

   cl_command_queue * getCommandQueue(){return & commands;}

   
//   int copyResultsBuffer(cl_mem output, void * results, size_t size);

   int query_device_info();
   int query_device_info(int id, cl_device_id device);
   size_t get_max_work_group();
   size_t get_max_work_item_dimension(int dimension);

protected:
   cl_uint num_devices;                  // number of computing devices

   cl_device_id device_ids[MAX_DEVICES]; // compute device id
   cl_context context;                   // compute context
   cl_command_queue commands;            // compute command queue
#endif
};

} // namespace PV

#endif /* CLDEVICE_HPP_ */
