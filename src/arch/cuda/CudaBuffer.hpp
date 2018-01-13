/*
 * CudaBuffer.hpp
 *
 *  Created on: Aug 6, 2014
 *      Author: Sheng Lundquist
 */

#ifndef CUDABUFFER_HPP_
#define CUDABUFFER_HPP_

#include <cuda_runtime_api.h>

////////////////////////////////////////////////////////////////////////////////

namespace PVCuda {

class CudaDevice;

/**
 * A class to handle device memory allocations and transfers
 */
class CudaBuffer {

  public:
   /**
    * Constructor to create a buffer of size inSize on a given stream
    * @param inSize The size of the buffer to create on the device
    * @param stream The cuda stream any transfer commands should go on
    */
   CudaBuffer(size_t inSize, CudaDevice *inDevice, cudaStream_t stream);
   CudaBuffer();
   virtual ~CudaBuffer();

   /**
    * A function to copy host memory to device memory. Note that the host and device memory must
    * have the same size, otherwise behavior is undefined.
    * @param h_ptr The pointer to the host address to copy to the device
    * #return Returns PV_Success if successful
    */
   int copyToDevice(const void *h_ptr);

   /**
    * A function to copy host memory to a portion of device memory.
    * offset plus in_size must not be greater than getSize().
    * Note that the offset is applied only to the device memory pointer,
    * not the host memory pointer h_ptr: h_ptr[0] through h_ptr[in_size-1]
    * are copied to d_ptr[offset] through d_ptr[offset + in_size - 1].
    * @param h_ptr The pointer to the host address to copy to the device
    * @param in_size The number of bytes for the data to copy
    * @param offset The offset into the device memory at which the copy should start.
    */
   int copyToDevice(const void *h_ptr, size_t in_size, size_t offset);

   /**
    * A function to copy device memory to host memory. Note that the host and device memory must
    * have the same size, otherwise undefined behavior
    * @param h_ptr The pointer to the host address to copy from the device
    * @param in_size The size of the data to copy. Defaults to the size of the buffer.
    * #return Returns PV_Success if successful
    */
   int copyFromDevice(void *h_ptr);
   int copyFromDevice(void *h_ptr, size_t in_size);

   /**
    * A getter function to return the device pointer allocated
    * #return Returns the device pointer
    */
   void *getPointer() { return d_ptr; }

   /**
    * A getter function to return the size of the device memory
    * #return Returns the size of the device memory
    */
   size_t getSize() { return size; }

   void
   permuteWeightsPVToCudnn(void *d_inPtr, int numArbors, int numKernels, int nxp, int nyp, int nfp);

  protected:
   void *d_ptr; // pointer to buffer on device
   size_t size;
   cudaStream_t stream;
   CudaDevice *device;

  private:
   void callCudaPermuteWeightsPVToCudnn(
         int gridSize,
         int blockSize,
         void *d_inPtr,
         int numArbors,
         int outFeatures,
         int ny,
         int nx,
         int inFeatures);
}; // end class CudaBuffer

} // end namespace PVCuda

#endif
