/*
 * CLDevice.cpp
 *
 *  Created on: Oct 24, 2009
 *      Author: rasmussn
 */

#include "CLDevice.hpp"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#ifdef PV_USE_OPENCL

namespace PV {

static char * load_program_source(const char *filename);
	
CLDevice::CLDevice(int device)
{
   this->device = device;
   initialize(device);
}

CLDevice::~CLDevice()
{
}

int CLDevice::initialize(int device)
{
   int status = 0;

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
   context = clCreateContext(0, 1, &device_ids[device], NULL, NULL, &status);
   if (!context)
   {
       printf("Error: Failed to create a compute context for device %d!\n", device);
       exit(PVCL_CREATE_CONTEXT_FAILURE);
   }

   // create a command queue
   //
   commands = clCreateCommandQueue(context, device_ids[device], 0, &status);
   if (!commands)
   {
       printf("Error: Failed to create a command commands!\n");
       return PVCL_CREATE_CMD_QUEUE_FAILURE;
   }

   // turn on profiling
   //
   elapsed = 0;
   profiling = true;
   status = clSetCommandQueueProperty(commands, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
   if (status != CL_SUCCESS) {
      print_error_code(status);
      exit(status);
   }

   return status;
}

int CLDevice::run(size_t global_work_size)
{
   size_t local_work_size;
   int status = 0;
		
   // get the maximum work group size for executing the kernel on the device
   //
   status = clGetKernelWorkGroupInfo(kernel, device_ids[device], CL_KERNEL_WORK_GROUP_SIZE,
						             sizeof(size_t), &local_work_size, NULL);
   if (status != CL_SUCCESS) {
      fprintf(stderr, "Error: Failed to retrieve kernel work group info! %d\n", status);
      print_error_code(status);
      exit(status);
   } else {
      printf("run: local_work_size==%ld global_work_size==%ld\n", local_work_size, global_work_size);
   }
   // execute the kernel over the entire range of our 1d input data set
   // using the maximum number of work group items for this device
   //
   status = clEnqueueNDRangeKernel(commands, kernel, 1, NULL,
								   &global_work_size, &local_work_size, 0, NULL, NULL);
   if (status != CL_SUCCESS) {
      fprintf(stderr, "CLDevice::run(): Failed to execute kernel!\n");
      print_error_code(status);
      exit(status);
   }
		
   // wait for the command commands to get serviced before reading back results
   //
   clFinish(commands);
		
   return status;
}
	
int CLDevice::run(size_t gWorkSizeX, size_t gWorkSizeY, size_t lWorkSizeX, size_t lWorkSizeY)
{
   size_t local_work_size[2];
   size_t global_work_size[2];
   size_t max_local_size;
   int status = 0;
		
   global_work_size[0] = gWorkSizeX;
   global_work_size[1] = gWorkSizeY;

   local_work_size[0] = lWorkSizeX;
   local_work_size[1] = lWorkSizeY;
	
   // get the maximum work group size for executing the kernel on the device
   //
   status = clGetKernelWorkGroupInfo(kernel, device_ids[device], CL_KERNEL_WORK_GROUP_SIZE,
								     sizeof(size_t), &max_local_size, NULL);
   if (status != CL_SUCCESS) {
      fprintf(stderr, "Error: Failed to retrieve kernel work group info! (status==%d)\n", status);
      print_error_code(status);
      exit(status);
   } else {
      printf("run: local_work_size==(%ld,%ld) global_work_size==(%ld,%ld)\n",
             local_work_size[0], local_work_size[1], global_work_size[0], global_work_size[1]);
   }

   if (device == 1) {
      // Apple's CPU device has only one thread per group
	  local_work_size[0] = 1;
      local_work_size[1] = 1;
   }
	
   // execute the kernel over the entire range of our 1d input data set
   // using the maximum number of work group items for this device
   //
   status = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
                                   global_work_size, local_work_size, 0, NULL, &event);
   if (status) {
      fprintf(stderr, "CLDevice::run(): Failed to execute kernel! (status==%d)\n", status);
      fprintf(stderr, "CLDevice::run(): max_local_work_size==%ld\n", max_local_size);
      print_error_code(status);
      exit(status);
   }
		
   // wait for the command commands to get serviced before reading back results
   //
   clFinish(commands);
		
   // get profiling information
   //
   if (profiling) {
      size_t param_size;
      cl_ulong start, end;
      status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                       sizeof(start), &start, &param_size);
      status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                       sizeof(end), &end, &param_size);
      if (status == 0) {
         elapsed = (end - start) / 1000;  // microseconds
      }
   }
   
   return status;
}
	
int CLDevice::createKernel(const char * filename, const char * name)
{
   int status = 0;

   // Create the compute program from the source buffer
   //
   char * source = load_program_source(filename);
   program = clCreateProgramWithSource(context, 1, (const char **) &source, NULL, &status);
   if (!program || status != CL_SUCCESS)
   {
       printf("Error: Failed to create compute program!\n");
       print_error_code(status);
       exit(status);
   }

   // Build the program executable
   //
   status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if (status != CL_SUCCESS)
   {
       size_t len;
       char buffer[2048];

       printf("Error: Failed to build program executable!\n");
       clGetProgramBuildInfo(program, device_ids[device], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
       printf("%s\n", buffer);
       print_error_code(status);
       exit(status);
   }

   // Create the compute kernel in the program we wish to run
   //
   kernel = clCreateKernel(program, name, &status);
   if (!kernel || status != CL_SUCCESS)
   {
       fprintf(stderr, "Error: Failed to create compute kernel!\n");
       print_error_code(status);
       exit(status);
   }

   return status;
}

CLBuffer * CLDevice::createBuffer(cl_mem_flags flags, size_t size, void * host_ptr)
{
   return new CLBuffer(context, commands, flags, size, host_ptr);
}

int CLDevice::addKernelArg(int argid, int arg)
{
   int status = 0;
      
   status = clSetKernelArg(kernel, argid, sizeof(int), &arg);
   if (status != CL_SUCCESS) {
      fprintf(stderr, "CLDevice::addKernelArg: Failed to set kernel argument! %d\n", status);
      print_error_code(status);
      exit(status);
   }
   
   return status;
}
   
int CLDevice::addKernelArg(int argid, CLBuffer * buf)
{
   int status = 0;

   cl_mem mobj = buf->clMemObject();
      
   status = clSetKernelArg(kernel, argid, sizeof(cl_mem), &mobj);
   if (status != CL_SUCCESS) {
      fprintf(stderr, "CLDevice::addKernelArg: Failed to set kernel argument! %d\n", status);
      print_error_code(status);
      exit(status);
   }
   
   return status;
}
   
int CLDevice::addLocalArg(int argid, size_t size)
{
   int status = 0;

   status = clSetKernelArg(kernel, argid, size, 0);
   if (status != CL_SUCCESS) {
      fprintf(stderr, "CLDevice::addLocalArg: Failed to set kernel argument! %d\n", status);
      print_error_code(status);
      exit(status);
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
   for (int i = 0; i < max_dims; i++) printf(" %ld", vals[i]);
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
print_error_code(int code)
{
   char msg[256];

   switch (code) {
      case CL_INVALID_WORK_GROUP_SIZE:
         sprintf(msg, "%s (%d)", "CL_INVALID_WORK_GROUP_SIZE", code);
         break;
      case CL_INVALID_COMMAND_QUEUE:
         sprintf(msg, "%s (%d)", "CL_INVALID_COMMAND_QUEUE", code);
         break;
      case CL_INVALID_KERNEL_ARGS:
         sprintf(msg, "%s (%d)", "CL_INVALID_KERNEL_ARGS", code);
         break;

      default:
         sprintf(msg, "%s (%d)\n", "UNKNOWN_CODE", code);
         break;
   }
   printf("ERROR_CODE==%s\n", msg);
}

static char *
load_program_source(const char *filename)
{
    struct stat statbuf;
    FILE        *fh;
    char        *source;

    fh = fopen(filename, "r");
    if (fh == 0)
        return 0;

    stat(filename, &statbuf);
    source = (char *) malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';

    return source;
}

} // namespace PV

#else
void cldevice_noop() { ; }
#endif // PV_USE_OPENCL
