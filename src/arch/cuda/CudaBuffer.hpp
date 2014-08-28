/*
 * CudaBuffer.hpp
 *
 *  Created on: Aug 6, 2014
 *      Author: Sheng Lundquist
 */

#ifndef CUDABUFFER_HPP_
#define CUDABUFFER_HPP_

////////////////////////////////////////////////////////////////////////////////

namespace PVCuda {

class CudaBuffer {

public:

   CudaBuffer(size_t inSize);
   CudaBuffer();
   virtual ~CudaBuffer();
   
   virtual int copyToDevice(void * h_ptr);
   virtual int copyFromDevice(void* h_ptr);

   virtual void* getPointer(){return d_ptr;}
   size_t getSize(){return size;}

protected:
   void * d_ptr;                       // pointer to buffer on host
   size_t size;
};

} // namespace PV

#endif
