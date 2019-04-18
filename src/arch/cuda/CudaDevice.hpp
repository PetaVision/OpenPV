/*
 * CudaDevice.hpp
 *
 *  Created on: July 30, 2014
 *      Author: Sheng Lundquist
 */

#ifndef CUDADEVICE_HPP_
#define CUDADEVICE_HPP_

#include "../../include/pv_arch.h"
#include "CudaBuffer.hpp"
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string>

namespace PVCuda {

/**
 * A class to handle initialization of cuda devices
 */
class CudaDevice {
  protected:
   int device_id; // device id (normally 0 for GPU, 1 for CPU)

  public:
   void incrementConvKernels();
   size_t getMemory() { return deviceMem; }
   size_t getNumConvKernels() { return numConvKernels; }

   static int getNumDevices();

   /**
    * A constructor to create the device object
    * @param device The device number to use
    */
   CudaDevice(int device);
   virtual ~CudaDevice();

   /**
    * A function to initialize the device
    * @param device The device number to initialize
    */
   int initialize(int device);

   /**
    * A getter function to return what device is being used
    * @return The device number of the device being used
    */
   int id() { return device_id; }

   /**
    * A function to create a buffer from the given stream
    * @param size The size of the buffer being created
    * @param str  A string used in the message logging the buffer creation.
    * @return The CudaBuffer object from creating the buffer
    */
   CudaBuffer *createBuffer(size_t size, std::string const *str);

   /**
    * A function to return the cuda stream the device is using
    * @return The stream the device is using
    */
   cudaStream_t getStream() { return stream; }

   /**
    * A synchronization barrier to block the cpu from running until the gpu stream has finished
    */
   void syncDevice();

   /**
    * A function to query all device information
    */
   int query_device_info();
   /**
    * A function to query a given device's information
    * @param id The device ID to get infromation from
    */
   void query_device(int id);

   /**
    * A getter function to return the max threads of the currently used device
    * @return The max number of threads on the device
    */
   int get_max_threads();

   /**
    * A getter function to return the max block size of a given dimension of the currently used
    * device
    * @param dimension The dimension of the block size. Has to be 0-2 inclusive
    * @return The max block size of the given dimension on the device
    */
   int get_max_block_size_dimension(int dimension);

   /**
    * A getter function to return the max grid size of a given dimension of the currently used
    * device
    * @param dimension The dimension of the grid size. Has to be 0-2 inclusive
    * @return The max grid size of the given dimension on the device
    */
   int get_max_grid_size_dimension(int dimension);

   /**
    * A getter function to return the warp size of the currently used device
    * @return The warp size of the device
    */
   int get_warp_size();

   /**
    * A getter function to return the local memory size of the currently used device
    * @return The local memory size of the device
    */
   size_t get_local_mem();

#ifdef PV_USE_CUDNN
   void *getCudnnHandle() { return handle; }
#endif

  private:
   /**
    * Decrements deviceMem by the given number of bytes, and exits with an error if deviceMem drops
    * below zero.
    * Called by createBuffer.
    */
   long reserveMem(size_t size);

  protected:
   int num_devices; // number of computing devices
   struct cudaDeviceProp device_props;
   cudaStream_t stream;
   long deviceMem;
   size_t numConvKernels = (size_t)0;

   void *handle;
};

} // namespace PV

#endif /* CLDEVICE_HPP_ */
