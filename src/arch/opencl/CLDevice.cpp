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
#include "CLTimer.hpp"
#include "utils/PVLog.hpp"
#include "utils/PVAssert.hpp"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

namespace PV {

#ifdef PV_USE_OPENCL

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

   // get list of available platforms
   //
   cl_uint num_platforms;
   cl_platform_id platforms[2];

   size_t name_size = 63;
   char platform_name[64];
   char errorString[256];
   errorString[0] = '\0';

   status = clGetPlatformIDs(0, NULL, &num_platforms);
   if (status != CL_SUCCESS) {
      print_error_code(status, errorString, 256);
      pvError() << "Failed to get number of available platforms!\n" << errorString;
   }


   status = clGetPlatformIDs(2, platforms, &num_platforms);
   if (status != CL_SUCCESS) {
      print_error_code(status, errorString, 256);
      pvError() << "Failed to get platform ids!\n" << errorString;
   }
   if (num_platforms > 1) {
      pvWarn().printf("Warning: number of platforms is %d\n", num_platforms);
   }

   // get info about the platform
   //
   status = clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, name_size, platform_name, &name_size);
   if (status != CL_SUCCESS) {
      print_error_code(status, errorString, 256);
      pvError() << "Failed to get platform info!\n" << errorString;
   }

   // get number of devices available
   //
   status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, MAX_DEVICES, device_ids, &num_devices);
   if (status != CL_SUCCESS) {
      print_error_code(status, errorString, 256);
      pvError() << "Failed to find a device group!\n" << errorString;
   }

   pvInfo().printf("Using device %d\n", device_id);

   // create a compute context
   //
   context = clCreateContext(0, 1, &device_ids[device_id], NULL, NULL, &status);
   if (!context)
   {
       pvError().printf("Failed to create a compute context for device %d!\n", device);
   }

   // create a command queue
   //
   commands = clCreateCommandQueue(context, device_ids[device_id], CL_QUEUE_PROFILING_ENABLE, &status);
   assert(status == CL_SUCCESS);
   if (!commands)
   {
       pvWarn().printf("Failed to create a command queue!\n");
       return PVCL_CREATE_CMD_QUEUE_FAILURE;
   }

   // turn on profiling for this command queue
   //
//   status = clSetCommandQueueProperty(commands, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
//   if (status != CL_SUCCESS) {
//      print_error_code(status, errorString, 256);
//      pvError() << errorString;
//   }
   status = 0;

   return status;
}

CLTimer* CLDevice::createTimer(double init_time){
   return new CLTimer(commands, init_time);
}
CLTimer* CLDevice::createTimer(const char * timermessage, double init_time){
   return new CLTimer(commands, timermessage, init_time);
}
CLTimer* CLDevice::createTimer(const char * objname, const char * objtype, const char * timertype, double init_time){
   return new CLTimer(commands, objname, objtype, timertype, init_time);
}

CLKernel * CLDevice::createKernel(const char * filename, const char * name, const char * options)
{
   return new CLKernel(context, commands, device_ids[device_id], filename, name, options);
}

CLBuffer * CLDevice::createBuffer(cl_mem_flags flags, size_t size, void * host_ptr)
{
   return new CLBuffer(context, commands, flags, size, host_ptr);
}

size_t CLDevice::get_max_work_group(){
   int    status;
   cl_device_id device = device_ids[device_id];
   size_t val;
   size_t param_value_size;
   status = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &val, &param_value_size);
   return val;
}

size_t CLDevice::get_max_work_item_dimension(int dimension){
   int    status;
   cl_device_id device = device_ids[device_id];
   const int vals_len = MAX_WORK_ITEM_DIMENSIONS;
   size_t vals[vals_len];
   if(dimension >= vals_len){
      return 0;
   }
   size_t param_value_size;
   status = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, vals_len*sizeof(size_t), vals, &param_value_size);
   return vals[dimension];
}


int CLDevice::syncDevice(){
   return clFinish(commands);
}


int CLDevice::query_device_info()
{
   // query and print information about the devices found
   //
   pvInfo().printf("\n");
   pvInfo().printf("Number of OpenCL devices found: %d\n", num_devices);
   pvInfo().printf("\n");

   for (int i = 0; i < num_devices; i++) {
      query_device_info(i, device_ids[i]);
   }
   return 0;
}




int CLDevice::query_device_info(int id, cl_device_id device)
{
   const int str_size = 200;
   const int vals_len = MAX_WORK_ITEM_DIMENSIONS;

   long long val;
   size_t vals[vals_len];
   unsigned int max_dims;

   int    status;
   char   param_value[str_size];
   size_t param_value_size;

   pvInfo().printf("device: %d\n", id);

   status = clGetDeviceInfo(device, CL_DEVICE_NAME, str_size, param_value, &param_value_size);
   param_value[str_size-1] = '\0';

   pvInfo().printf("OpenCL Device # %d == %s\n", id, param_value);

   status = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(val), &val, NULL);

   if (status == CL_SUCCESS) {
      pvInfo().printf("\tdevice[%p]: Type: ", device);

      if (val & CL_DEVICE_TYPE_DEFAULT) {
         val &= ~CL_DEVICE_TYPE_DEFAULT;
         pvInfo().printf("Default ");
      }

      if (val & CL_DEVICE_TYPE_CPU) {
         val &= ~CL_DEVICE_TYPE_CPU;
         pvInfo().printf("CPU ");
      }

      if (val & CL_DEVICE_TYPE_GPU) {
         val &= ~CL_DEVICE_TYPE_GPU;
         pvInfo().printf("GPU ");
      }

      if (val & CL_DEVICE_TYPE_ACCELERATOR) {
         val &= ~CL_DEVICE_TYPE_ACCELERATOR;
         pvInfo().printf("Accelerator ");
      }

      if (val != 0) {
         pvInfo().printf("Unknown (0x%llx) ", val);
      }
   }
   else {
      char errorString[256];
      print_error_code(status, errorString, 256);
      pvError(errorMessage);
      errorMessage.printf("\tdevice[%p]: Unable to get TYPE: %s!\n", device, "CLErrString(status)");
      errorMessage << errorString;
   }

   status = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(val), &val, &param_value_size);
   pvInfo().printf("with %u units/cores", (unsigned int) val);

   status = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(val), &val, &param_value_size);
   pvInfo().printf(" at %u MHz\n", (unsigned int) val);

   status = clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(val), &val, &param_value_size);
   pvInfo().printf("\tfloat vector width == %u\n", (unsigned int) val);
   
   status = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(val), &val, &param_value_size);
   pvInfo().printf("\tMaximum work group size == %lu\n", (size_t) val);
   
   status = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_dims), &max_dims, &param_value_size);
   pvInfo().printf("\tMaximum work item dimensions == %u\n", max_dims);
   
   status = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, vals_len*sizeof(size_t), vals, &param_value_size);
   pvInfo().printf("\tMaximum work item sizes == (");
   for (unsigned int i = 0; i < max_dims; i++) pvInfo().printf(" %ld", vals[i]);
   pvInfo().printf(" )\n");
   
   status = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(val), &val, &param_value_size);
   pvInfo().printf("\tLocal mem size == %u\n", (unsigned int) val);

   status = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(val), &val, &param_value_size);
   pvInfo().printf("\tGlobal mem size == %u\n", (unsigned int) val);

   status = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(val), &val, &param_value_size);
   pvInfo().printf("\tGlobal mem cache size == %u\n", (unsigned int) val);

   status = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(val), &val, &param_value_size);
   pvInfo().printf("\tGlobal mem cache line size == %u\n", (unsigned int) val);

   status = clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(val), &val, &param_value_size);
   pvInfo().printf("\tMax constant arguments == %u\n", (unsigned int) val);

   status = clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(val), &val, &param_value_size);
   pvInfo().printf("\tMax constant buffer size == %lu\n", (unsigned long) val);

   status = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(val), &val, &param_value_size);
   pvInfo().printf("\tMax mem alloc size == %lu\n", (unsigned long) val);

   status = clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(val), &val, &param_value_size);
   pvInfo().printf("\tPreferred vector width float == %u\n", (unsigned int) val);

   pvInfo().printf("\n");

   return status;
}

/////////////////////////////////////////////////////////////////////////////
	
void
CLDevice::print_error_code(int code, char * buffer, int n)
{
   switch (code) {
      case CL_SUCCESS:
         return;
      case CL_INVALID_ARG_INDEX:
         snprintf(buffer, n, "%s (%d)", "CL_INVALID_ARG_INDEX", code);
         break;
      case CL_INVALID_ARG_SIZE:
         snprintf(buffer, n, "%s (%d)", "CL_INVALID_ARG_SIZE", code);
         break;
      case CL_INVALID_ARG_VALUE:
         snprintf(buffer, n, "%s (%d)", "CL_INVALID_ARG_VALUE", code);
         break;
      case CL_INVALID_BUFFER_SIZE:
         snprintf(buffer, n, "%s (%d)", "CL_INVALID_BUFFER_SIZE", code);
         break;
      case CL_INVALID_COMMAND_QUEUE:
         snprintf(buffer, n, "%s (%d)", "CL_INVALID_COMMAND_QUEUE", code);
         break;
      case CL_INVALID_CONTEXT:
         snprintf(buffer, n, "%s (%d)", "CL_INVALID_CONTEXT", code);
         break;
      case CL_INVALID_EVENT_WAIT_LIST:
         snprintf(buffer, n, "%s (%d)", "CL_INVALID_EVENT_WAIT_LIST", code);
         break;
      case CL_INVALID_KERNEL_NAME:
         snprintf(buffer, n, "%s (%d)", "CL_INVALID_KERNEL_NAME", code);
         break;
      case CL_INVALID_MEM_OBJECT:
         snprintf(buffer, n, "%s (%d)", "CL_INVALID_MEM_OBJECT", code);
         break;
      case CL_INVALID_PROGRAM_EXECUTABLE:
         snprintf(buffer, n, "%s (%d)", "CL_INVALID_PROGRAM_EXECUTABLE", code);
         break;
      case CL_BUILD_PROGRAM_FAILURE:
         snprintf(buffer, n, "%s (%d)", "CL_BUILD_PROGRAM_FAILURE", code);
         break;
      case CL_INVALID_HOST_PTR:
         snprintf(buffer, n, "%s (%d)", "CL_INVALID_HOST_PTR", code);
         break;
      case CL_INVALID_KERNEL_ARGS:
         snprintf(buffer, n, "%s (%d)", "CL_INVALID_KERNEL_ARGS", code);
         break;
      case CL_INVALID_PLATFORM:
         snprintf(buffer, n, "%s (%d)", "CL_INVALID_PLATFORM", code);
         break;
      case CL_INVALID_QUEUE_PROPERTIES:
         snprintf(buffer, n, "%s (%d)", "CL_INVALID_QUEUE_PROPERTIES", code);
         break;
      case CL_INVALID_VALUE:
         snprintf(buffer, n, "%s (%d)", "CL_INVALID_VALUE", code);
         break;
      case CL_INVALID_WORK_GROUP_SIZE:
         snprintf(buffer, n, "%s (%d)", "CL_INVALID_WORK_GROUP_SIZE", code);
         break;
      case CL_OUT_OF_HOST_MEMORY:
         snprintf(buffer, n, "%s (%d)", "CL_OUT_HOST_MEMORY", code);
         break;
      case CL_OUT_OF_RESOURCES:
         snprintf(buffer, n, "%s (%d)", "CL_OUT_OF_RESOURCES", code);
         break;
      case CL_PROFILING_INFO_NOT_AVAILABLE:
         snprintf(buffer, n, "%s (%d)", "CL_PROFILING_INFO_NOT_AVAILABLE", code);
         break;
      default:
         snprintf(buffer, n, "%s (%d)\n", "UNKNOWN_CODE", code);
         break;
  }
}

#endif // PV_USE_OPENCL

}



 // namespace PV

