/*
 * CLKernel.cpp
 *
 *  Created on: Aug 1, 2010
 *      Author: rasmussn
 */

#include "CLKernel.hpp"
#include "CLDevice.hpp"
#include <stdio.h>
#include <sys/stat.h>

namespace PV {

static char * load_program_source(const char *filename);

CLKernel::CLKernel(cl_context context, cl_command_queue commands, cl_device_id device,
                   const char * filename, const char * name)
{
   this->device = device;
   this->commands = commands;
   this->profiling = true;
   this->elapsed = 0;

#ifdef PV_USE_OPENCL

   int status = CL_SUCCESS;

   // Create the compute program from the source buffer
   //
   char * source = load_program_source(filename);
   program = clCreateProgramWithSource(context, 1, (const char **) &source, NULL, &status);
   if (!program || status != CL_SUCCESS)
   {
       printf("Error: Failed to create compute program!\n");
       CLDevice::print_error_code(status);
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
       clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
       printf("%s\n", buffer);
       CLDevice::print_error_code(status);
       exit(status);
   }

   // Create the compute kernel in the program we wish to run
   //
   kernel = clCreateKernel(program, name, &status);
   if (!kernel || status != CL_SUCCESS)
   {
       fprintf(stderr, "Error: Failed to create compute kernel!\n");
       CLDevice::print_error_code(status);
       exit(status);
   }

#endif // PV_USE_OPENCL

}

int CLKernel::run(size_t global_work_size)
{
   int status = CL_SUCCESS;

#ifdef PV_USE_OPENCL

   size_t local_work_size;

   // get the maximum work group size for executing the kernel on the device
   //
   status = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                     sizeof(size_t), &local_work_size, NULL);
   if (status != CL_SUCCESS) {
      fprintf(stderr, "Error: Failed to retrieve kernel work group info! %d\n", status);
      CLDevice::print_error_code(status);
      exit(status);
   } else {
      printf("run: local_work_size==%ld global_work_size==%ld\n", local_work_size, global_work_size);
   }
   // execute the kernel over the entire range of our 1d input data set
   // using the maximum number of work group items for this device
   //
   status = clEnqueueNDRangeKernel(commands, kernel, 1, NULL,
                                   &global_work_size, &local_work_size, 0, NULL, &event);
   if (status != CL_SUCCESS) {
      fprintf(stderr, "CLDevice::run(): Failed to execute kernel!\n");
      CLDevice::print_error_code(status);
      exit(status);
   }

   // wait for the command commands to get serviced before reading back results
   //
   clFinish(commands);

#endif // PV_USE_OPENCL

   return status;
}

int CLKernel::run(size_t gWorkSizeX, size_t gWorkSizeY, size_t lWorkSizeX, size_t lWorkSizeY)
{
   int status = CL_SUCCESS;

#ifdef PV_USE_OPENCL

   size_t local_work_size[2];
   size_t global_work_size[2];
   size_t max_local_size;

   global_work_size[0] = gWorkSizeX;
   global_work_size[1] = gWorkSizeY;

   local_work_size[0] = lWorkSizeX;
   local_work_size[1] = lWorkSizeY;

#ifdef PV_USE_TAU
   int tau_id = 10;
   TAU_START("CLKernel::run::CPU");
#endif

   // get the maximum work group size for executing the kernel on the device
   //
   status = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                     sizeof(size_t), &max_local_size, NULL);
   if (status != CL_SUCCESS) {
      fprintf(stderr, "Error: Failed to retrieve kernel work group info! (status==%d)\n", status);
      CLDevice::print_error_code(status);
      exit(status);
   } else {
      //printf("run: local_work_size==(%ld,%ld) global_work_size==(%ld,%ld)\n",
      //       local_work_size[0], local_work_size[1], global_work_size[0], global_work_size[1]);
   }

   // execute the kernel over the entire range of our 1d input data set
   // using the maximum number of work group items for this device
   //
   status = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
                                   global_work_size, local_work_size, 0, NULL, &event);
   if (status) {
      fprintf(stderr, "CLDevice::run(): Failed to execute kernel! (status==%d)\n", status);
      fprintf(stderr, "CLDevice::run(): max_local_work_size==%ld\n", max_local_size);
      CLDevice::print_error_code(status);
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
#ifdef PV_USE_TAU
      tau_id += 1000;
      TAU_STOP("CLKernel::run::CPU");
#endif
      status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                       sizeof(start), &start, &param_size);
      status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                       sizeof(end), &end, &param_size);
      if (status == 0) {
         elapsed = (end - start) / 1000;  // microseconds
      }
#ifdef PV_USE_TAU
      Tau_opencl_register_gpu_event("CLKernel::run::GPU", tau_id, start, end);
#endif
   }

#endif // PV_USE_OPENCL

   return status;
}

int CLKernel::addKernelArg(int argid, int arg)
{
   int status = CL_SUCCESS;

#ifdef PV_USE_OPENCL

   status = clSetKernelArg(kernel, argid, sizeof(int), &arg);
   if (status != CL_SUCCESS) {
      fprintf(stderr, "CLDevice::addKernelArg: Failed to set kernel argument! %d\n", status);
      CLDevice::print_error_code(status);
      exit(status);
   }

#endif // PV_USE_OPENCL

   return status;
}

int CLKernel::addKernelArg(int argid, CLBuffer * buf)
{
   int status = CL_SUCCESS;

#ifdef PV_USE_OPENCL
   cl_mem mobj = buf->clMemObject();

   status = clSetKernelArg(kernel, argid, sizeof(cl_mem), &mobj);
   if (status != CL_SUCCESS) {
      fprintf(stderr, "CLDevice::addKernelArg: Failed to set kernel argument! %d\n", status);
      CLDevice::print_error_code(status);
      exit(status);
   }
#endif // PV_USE_OPENCL

   return status;
}

int CLKernel::addLocalArg(int argid, size_t size)
{
   int status = CL_SUCCESS;

#ifdef PV_USE_OPENCL

   status = clSetKernelArg(kernel, argid, size, 0);
   if (status != CL_SUCCESS) {
      fprintf(stderr, "CLDevice::addLocalArg: Failed to set kernel argument! %d\n", status);
      CLDevice::print_error_code(status);
      exit(status);
   }

#endif // PV_USE_OPENCL

   return status;
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

}
