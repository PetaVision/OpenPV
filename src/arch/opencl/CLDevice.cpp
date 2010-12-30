/*
 * CLDevice.cpp
 *
 *  Created on: Oct 24, 2009
 *      Author: Craig Rasmussen
 */

// NOTE: order must be preserved for first
// two includes
#include "../../include/pv_arch.h"
#include "pv_opencl.h"

#include "CLDevice.hpp"
#include "CLBuffer.hpp"
#include "CLKernel.hpp"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

namespace PV {

CLDevice::CLDevice(int device)
{
   this->device_id = device;
   initialize(device_id);

   // initialize TAU profiling
   //
#ifdef PV_USE_TAU
   Tau_opencl_init();
#endif
}

CLDevice::~CLDevice()
{
   // finalize TAU profiling
   //
#ifdef PV_USE_TAU
   Tau_opencl_exit();
#endif
}

int CLDevice::initialize(int device)
{
   int status = 0;

#ifdef PV_USE_OPENCL
   // get number of devices available
   //
   status = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, MAX_DEVICES, device_ids, &num_devices);
   if (status != CL_SUCCESS) {
      printf("Error: Failed to find a device group!\n");
      print_error_code(status);
      exit(status);
   }

   // create a compute context
   //
   context = clCreateContext(0, 1, &device_ids[device_id], NULL, NULL, &status);
   if (!context)
   {
       printf("Error: Failed to create a compute context for device %d!\n", device);
       exit(PVCL_CREATE_CONTEXT_FAILURE);
   }

   // create a command queue
   //
   commands = clCreateCommandQueue(context, device_ids[device_id], 0, &status);
   if (!commands)
   {
       printf("Error: Failed to create a command commands!\n");
       return PVCL_CREATE_CMD_QUEUE_FAILURE;
   }

   // turn on profiling for this command queue
   //
   status = clSetCommandQueueProperty(commands, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
   if (status != CL_SUCCESS) {
      print_error_code(status);
      exit(status);
   }
#endif PV_USE_OPENCL

   return status;
}

CLKernel * CLDevice::createKernel(const char * filename, const char * name)
{
   return new CLKernel(context, commands, device_ids[device_id], filename, name);
}

CLBuffer * CLDevice::createBuffer(cl_mem_flags flags, size_t size, void * host_ptr)
{
#ifdef PV_USE_OPENCL
   return new CLBuffer(context, commands, flags, size, host_ptr);
#else
   return new CLBuffer();
#endif
}

#ifdef PV_USE_OPENCL

int CLDevice::query_device_info()
{
   // query and print information about the devices found
   //
   printf("\n");
   printf("Number of OpenCL devices found: %d\n", num_devices);
   printf("\n");

   for (unsigned int i = 0; i < num_devices; i++) {
      query_device_info(i, device_ids[i]);
   }
   return 0;
}

int CLDevice::query_device_info(int id, cl_device_id device)
{
   const int str_size = 64;
   const int vals_len = MAX_WORK_ITEM_DIMENSIONS;

   long long val;
   size_t vals[vals_len];
   unsigned int max_dims;

   int    status;
   char   param_value[str_size];
   size_t param_value_size;

   status = clGetDeviceInfo(device, CL_DEVICE_NAME, str_size, param_value, &param_value_size);
   param_value[str_size-1] = '\0';

   printf("OpenCL Device # %d == %s\n", id, param_value);

   status = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(val), &val, NULL);

   if (status == CL_SUCCESS) {
      printf("\tdevice[%p]: Type: ", device);

      if (val & CL_DEVICE_TYPE_DEFAULT) {
         val &= ~CL_DEVICE_TYPE_DEFAULT;
         printf("Default ");
      }

      if (val & CL_DEVICE_TYPE_CPU) {
         val &= ~CL_DEVICE_TYPE_CPU;
         printf("CPU ");
      }

      if (val & CL_DEVICE_TYPE_GPU) {
         val &= ~CL_DEVICE_TYPE_GPU;
         printf("GPU ");
      }

      if (val & CL_DEVICE_TYPE_ACCELERATOR) {
         val &= ~CL_DEVICE_TYPE_ACCELERATOR;
         printf("Accelerator ");
      }

      if (val != 0) {
         printf("Unknown (0x%llx) ", val);
      }
   }
   else {
      printf("\tdevice[%p]: Unable to get TYPE: %s!\n", device, "CLErrString(status)");
      print_error_code(status);
      exit(status);
   }

   status = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(val), &val, &param_value_size);
   printf("with %u units/cores", (unsigned int) val);

   status = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(val), &val, &param_value_size);
   printf(" at %u MHz\n", (unsigned int) val);

   status = clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(val), &val, &param_value_size);
   printf("\tfloat vector width == %u\n", (unsigned int) val);
   
   status = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(val), &val, &param_value_size);
   printf("\tMaximum work group size == %lu\n", (size_t) val);
   
   status = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_dims), &max_dims, &param_value_size);
   printf("\tMaximum work item dimensions == %u\n", max_dims);
   
   status = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, vals_len*sizeof(size_t), vals, &param_value_size);
   printf("\tMaximum work item sizes == (");
   for (unsigned int i = 0; i < max_dims; i++) printf(" %ld", vals[i]);
   printf(" )\n");
   
   status = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(val), &val, &param_value_size);
   printf("\tLocal mem size == %u\n", (unsigned int) val);

   status = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(val), &val, &param_value_size);
   printf("\tGlobal mem size == %u\n", (unsigned int) val);

   status = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(val), &val, &param_value_size);
   printf("\tGlobal mem cache size == %u\n", (unsigned int) val);

   status = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(val), &val, &param_value_size);
   printf("\tGlobal mem cache line size == %u\n", (unsigned int) val);

   printf("\n");

   return status;
}

/////////////////////////////////////////////////////////////////////////////
	
void
CLDevice::print_error_code(int code)
{
   char msg[256];

   switch (code) {
      case CL_INVALID_WORK_GROUP_SIZE:
         sprintf(msg, "%s (%d)", "CL_INVALID_WORK_GROUP_SIZE", code);
         break;
      case CL_INVALID_COMMAND_QUEUE:
         sprintf(msg, "%s (%d)", "CL_INVALID_COMMAND_QUEUE", code);
         break;
      case CL_BUILD_PROGRAM_FAILURE:
         sprintf(msg, "%s (%d)", "CL_BUILD_PROGRAM_FAILURE", code);
         break;
      case CL_INVALID_HOST_PTR:
         sprintf(msg, "%s (%d)", "CL_INVALID_HOST_PTR", code);
         break;
      case CL_INVALID_KERNEL_ARGS:
         sprintf(msg, "%s (%d)", "CL_INVALID_KERNEL_ARGS", code);
         break;
      case CL_INVALID_VALUE:
         sprintf(msg, "%s (%d)", "CL_INVALID_VALUE", code);
         break;
      default:
         sprintf(msg, "%s (%d)\n", "UNKNOWN_CODE", code);
         break;
   }
   printf("ERROR_CODE==%s\n", msg);
}

#endif // PV_USE_OPENCL

} // namespace PV

