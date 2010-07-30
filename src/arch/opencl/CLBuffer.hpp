/*
 * CLBuffer.hpp
 *
 *  Created on: July 28, 2010
 *      Author: Craig Rasmussen
 */

#ifndef CLBUFFER_HPP_
#define CLBUFFER_HPP_

#include "../../include/pv_arch.h"

#ifdef PV_USE_OPENCL

#include <OpenCL/opencl.h>

////////////////////////////////////////////////////////////////////////////////

namespace PV {

class CLBuffer {
public:

   CLBuffer(cl_context context, cl_command_queue commands,
            cl_map_flags flags, size_t size, void * host_ptr);
   virtual ~CLBuffer();
   
   int copyToDevice  (void * host_ptr);
   int copyFromDevice(void * host_ptr);

   void * map(cl_map_flags flags);
   int    unmap(void * mapped_ptr);
   
   cl_mem clMemObject(void)   {return d_buf;}

protected:

   cl_command_queue commands;          // compute command queue

   size_t size;                        // size of buffer object
   cl_mem d_buf;                       // handle to buffer on the device
};

} // namespace PV

#endif /* PV_USE_OPENCL */
#endif /* CLBUFFER_HPP_ */
