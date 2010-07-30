#include "CLBuffer.hpp"
#include "CLDevice.hpp"

#include <stdio.h>
#include <stdlib.h>

#ifdef PV_USE_OPENCL

namespace PV {

CLBuffer::CLBuffer(cl_context context, cl_command_queue commands,
                   cl_map_flags flags, size_t size, void * host_ptr)
{
   int status = 0;

   this->commands = commands;
   this->size = size;

   this->d_buf = clCreateBuffer(context, flags, size, host_ptr, &status);
   if (status != CL_SUCCESS)
   {
      fprintf(stderr, "CLBuffer::CLBuffer: Failed to create buffer!\n");
      print_error_code(status);
      exit(1);
   }
}

CLBuffer::~CLBuffer()
{
   clReleaseMemObject(d_buf);
}
   
int CLBuffer::copyToDevice(void * host_ptr)
{
   int status = 0;
      
   // write data from host_ptr into the buffer in device memory
   //
   status = clEnqueueWriteBuffer(commands, d_buf, CL_TRUE, 0, size, host_ptr, 0, NULL, NULL);
   if (status != CL_SUCCESS)
   {
      fprintf(stderr, "CLBuffer::copyToDevice: Failed to enqueue write buffer!\n");
      print_error_code(status);
      exit(status);
   }
      
   return status;
}
   
int CLBuffer::copyFromDevice(void * host_ptr)
{
   int status = 0;
      
   // write data from host_ptr into the buffer in device memory
   //
   status = clEnqueueReadBuffer(commands, d_buf, CL_TRUE, 0, size, host_ptr, 0, NULL, NULL);
   if (status != CL_SUCCESS)
   {
      fprintf(stderr, "CLBuffer::copyFromDevice: Failed to enqueue read buffer!\n");
      print_error_code(status);
      exit(status);
   }
      
   return status;
}
   
void * CLBuffer::map(cl_map_flags flags)
{
   void * h_ptr;
   int status = 0;

   h_ptr = clEnqueueMapBuffer(commands, d_buf, CL_TRUE, flags, 0, size, 0, NULL, NULL, &status);
   if (status != CL_SUCCESS)
   {
      fprintf(stderr, "CLBuffer::map: Failed to enqueue map buffer!\n");
      print_error_code(status);
      exit(1);
   }

   return h_ptr;
}

int CLBuffer::unmap(void * mapped_ptr)
{
   int status = 0;

   status = clEnqueueUnmapMemObject(commands, d_buf, mapped_ptr, 0, NULL, NULL);
   if (status != CL_SUCCESS)
   {
      fprintf(stderr, "CLBuffer::unmap: Failed to enqueue unmap memory object!\n");
      print_error_code(status);
      exit(1);
   }

   return status;
}
   
} // namespace PV

#else
void clbuffer_noop() { ; }
#endif // PV_USE_OPENCL
