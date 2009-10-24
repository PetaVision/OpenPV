/*
 * CLDevice.cpp
 *
 *  Created on: Oct 24, 2009
 *      Author: rasmussn
 */

#include "CLDevice.hpp"

#include <stdio.h>

namespace PV {

CLDevice::CLDevice()
{
    initialize();
}

CLDevice::~CLDevice()
{
}

int CLDevice::initialize()
{
   int status = 0;

   // get number of devices available
   //
   status = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, MAX_DEVICES, device_ids, &num_devices);
   if (status != CL_SUCCESS) {
       printf("Error: Failed to find a device group!\n");
       return CL_GET_DEVICE_ID_FAILURE;
   }

   return status;
}

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

   long long val;

   int    status;
   char   param_value[str_size];
   size_t param_value_size;

   status = clGetDeviceInfo(device, CL_DEVICE_NAME, str_size, param_value, &param_value_size);
   param_value[str_size-1] = '\0';

   printf("OpenCL Device # %d == %s\n", id, param_value);

   status = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof val, &val, NULL);

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
   }

   status = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(val), &val, &param_value_size);
   printf("with %u units/cores", (unsigned int) val);

   status = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(val), &val, &param_value_size);
   printf(" at %u MHz\n", (unsigned int) val);

   status = clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(val), &val, &param_value_size);
   printf("\tfloat vector width == %u\n", (unsigned int) val);

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

} // namespace PV
