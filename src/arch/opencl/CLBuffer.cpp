/*
 * CLBuffer.cpp
 *
 *  Created on: Aug 1, 2010
 *      Author: Craig Rasmussen
 */

// NOTE: order must be preserved for first
// two includes
#include "../../include/pv_arch.h"
#include "pv_opencl.h"

#include "CLBuffer.hpp"
#include "CLDevice.hpp"
#include "utils/PVLog.hpp"
#include "utils/PVAssert.hpp"

#include <stdio.h>
#include <stdlib.h>

#ifdef PV_USE_OPENCL

namespace PV {

CLBuffer::CLBuffer(cl_context context, cl_command_queue commands,
                   cl_map_flags flags, size_t size, void * host_ptr)
{
   int status = 0;

   this->commands = commands;
   this->event = NULL;
   this->size = size;
   this->h_ptr = host_ptr;

   this->d_buf = clCreateBuffer(context, flags, size, host_ptr, &status);
   if (status != CL_SUCCESS)
   {
      char errorString[256];
      CLDevice::print_error_code(status, errorString, 256);
      pvError(errorMessage);
      errorMessage.printf("CLBuffer::CLBuffer: Failed to create buffer!\n");
      errorMessage << errorString;
   }

   this->profiling = false;
#ifdef PV_USE_TAU
   this->profiling = true;
#endif

}

CLBuffer::~CLBuffer()
{
   clReleaseMemObject(d_buf);
}
   
int CLBuffer::copyToDevice(void * host_ptr, unsigned int nWait, cl_event * waitList, cl_event * ev)
{
   int status = 0;

#ifdef PV_USE_TAU
   int tau_id = 0;
   Tau_opencl_enter_memcpy_event(tau_id, MemcpyHtoD);
#endif
      
   // write data from host_ptr into the buffer in device memory
   //
   status = clEnqueueWriteBuffer(commands, d_buf, CL_FALSE, 0, size,
                                 host_ptr, nWait, waitList, ev);

   //clFinish(commands);
#ifdef PV_USE_TAU
   Tau_opencl_exit_memcpy_event(tau_id, MemcpyHtoD);
#endif

   if (status != CL_SUCCESS)
   {
      char errorString[256];
      CLDevice::print_error_code(status, errorString, 256);
      pvError(errorMessage);
      errorMessage.printf("CLBuffer::copyToDevice: Failed to enqueue write buffer!\n");
      errorMessage << errorString;
   }
      
   return status;
}
   
int CLBuffer::copyFromDevice(void * host_ptr, unsigned int nWait, cl_event * waitList, cl_event * ev)
{
   int status = 0;
      
#ifdef PV_USE_TAU
   int tau_id = 1;
   Tau_opencl_enter_memcpy_event(tau_id, MemcpyDtoH);
#endif

   // write data from host_ptr into the buffer in device memory
   // TODO make this read a nonblocking read false
   //
   status = clEnqueueReadBuffer(commands, d_buf, CL_FALSE, 0, size,
                                host_ptr, nWait, waitList, ev);

#ifdef PV_USE_TAU
   Tau_opencl_exit_memcpy_event(tau_id, MemcpyDtoH);
#endif

   if (status != CL_SUCCESS)
   {
      char errorString[256];
      CLDevice::print_error_code(status, errorString, 256);
      pvError(errorMessage);
      errorMessage.printf("CLBuffer::copyFromDevice: Failed to enqueue read buffer!\n");
      errorMessage << errorString;
   }
      
   return status;
}
   
void * CLBuffer::map(cl_map_flags flags)
{
   //TODO doesn't work on neuro
   pvError().printf("Unmap not implemented\n");

   int status = 0;

#ifdef PV_USE_TAU
   int tau_id = 2;
   Tau_opencl_enter_memcpy_event(tau_id, MemcpyDtoH);
#endif

   h_ptr = clEnqueueMapBuffer(commands, d_buf, CL_FALSE, flags, 0, size, 0, NULL, &event, &status);

   if (status != CL_SUCCESS)
   {
      h_ptr = NULL;
      char errorString[256];
      CLDevice::print_error_code(status, errorString, 256);
      pvError(errorMessage);
      errorMessage.printf("CLBuffer::map: Failed to enqueue map buffer!\n");
      errorMessage << errorString;
   }

   //TODO - or use Marker?
   
   //TODO this doesn't work on neuro
   //status = clEnqueueBarrierWithWaitList(commands, 0, NULL, &event);

#ifdef PV_USE_TAU
   Tau_opencl_exit_memcpy_event(tau_id, MemcpyDtoH);
#endif
   if (status != CL_SUCCESS)
   {
      h_ptr = NULL;
      char errorString[256];
      CLDevice::print_error_code(status, errorString, 256);
      pvError(errorMessage);
      errorMessage.printf("CLBuffer::map: Failed in wait for event!\n");
      errorMessage << errorString;
   }

   // get profiling information
   //
   if (profiling) {
      size_t param_size;
      cl_ulong start, end;
      status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                       sizeof(start), &start, &param_size);
      status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                       sizeof(end), &end, &param_size);
#ifdef PV_USE_TAU
      tau_id += 1000;
      Tau_opencl_register_memcpy_event(tau_id, start, end, size, MemcpyDtoH);
#endif
   }

   return h_ptr;
}

int CLBuffer::unmap(void)
{
   return unmap(h_ptr);
}

int CLBuffer::unmap(void * mapped_ptr)
{
   int status = CL_SUCCESS;

   //TODO doesn't work on neuro
   pvError().printf("Unmap not implemented\n");

   h_ptr = NULL;  // buffer no longer mapped for host usage

#ifdef PV_USE_TAU
   int tau_id = 3;
   Tau_opencl_enter_memcpy_event(tau_id, MemcpyHtoD);
#endif

   status = clEnqueueUnmapMemObject(commands, d_buf, mapped_ptr, 0, NULL, &event);

   if (status != CL_SUCCESS)
   {
      char errorString[256];
      CLDevice::print_error_code(status, errorString, 256);
      pvError(errorMessage);
      errorMessage.printf("CLBuffer::unmap: Failed to enqueue unmap memory object!\n");
      errorMessage << errorString;
   }

   //TODO - or use Marker?
   //Doesn't work on neuro
   //status = clEnqueueBarrierWithWaitList(commands, 0, NULL, &event);


#ifdef PV_USE_TAU
   Tau_opencl_exit_memcpy_event(tau_id, MemcpyHtoD);
#endif
   if (status != CL_SUCCESS)
   {
      char errorString[256];
      CLDevice::print_error_code(status, errorString, 256);
      pvError(errorMessage);
      errorMessage.printf("CLBuffer::unmap: Failed in wait for event!\n");
      errorMessage << errorString;
   }

   // get profiling information
   //
   if (profiling) {
      size_t param_size;
      cl_ulong start, end;
      status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                       sizeof(start), &start, &param_size);
      status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                       sizeof(end), &end, &param_size);
#ifdef PV_USE_TAU
      tau_id += 1000;
      Tau_opencl_register_memcpy_event(tau_id, start, end, size, MemcpyHtoD);
#endif
   }

   return status;
}
   
} // namespace PV

#else
void clbuffer_noop() { ; }
#endif // PV_USE_OPENCL
