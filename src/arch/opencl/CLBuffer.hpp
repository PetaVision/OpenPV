/*
 * CLBuffer.hpp
 *
 *  Created on: July 28, 2010
 *      Author: Craig Rasmussen
 */

#ifndef CLBUFFER_HPP_
#define CLBUFFER_HPP_

#include "pv_opencl.h"

////////////////////////////////////////////////////////////////////////////////

namespace PV {

class CLBuffer {

#ifdef PV_USE_OPENCL

public:

   CLBuffer(cl_context context, cl_command_queue commands,
            cl_map_flags flags, size_t size, void * host_ptr);
   virtual ~CLBuffer();
   
   int copyToDevice  (void * host_ptr, unsigned int nWait, cl_event * evList, cl_event * ev);
   int copyFromDevice(void * host_ptr, unsigned int nWait, cl_event * evList, cl_event * ev);

   int copyToDevice  (cl_event * ev)  {return copyToDevice  (h_ptr, 0, NULL, ev);}
   int copyFromDevice(cl_event * ev)  {return copyFromDevice(h_ptr, 0, NULL, ev);}

   int copyToDevice(unsigned int nWait, cl_event * evList, cl_event * ev)
      {
         return copyToDevice(h_ptr, nWait, evList, ev);
      }
   int copyFromDevice(unsigned int nWait, cl_event * evList, cl_event * ev)
      {
         return copyFromDevice(h_ptr, nWait, evList, ev);
      }

   void * map(cl_map_flags flags);
   int  unmap(void * mapped_ptr);
   int  unmap(void);
   
   cl_mem clMemObject(void)   {return d_buf;}

protected:

   cl_command_queue commands;          // compute command queue
   cl_event event;                     // event identifying the kernel execution instance

   bool mapped;                        // true when buffer is mapped
   bool profiling;                     // flag to enable profiling

   size_t size;                        // size of buffer object
   cl_mem d_buf;                       // handle to buffer on the device
   void * h_ptr;                       // pointer to buffer on host

private: CLBuffer()    { ; }
#else
public:  CLBuffer()    { ; }
#endif /* PV_USE_OPENCL */

};

} // namespace PV

#endif /* CLBUFFER_HPP_ */
