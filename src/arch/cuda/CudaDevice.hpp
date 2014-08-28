/*
 * CudaDevice.hpp
 *
 *  Created on: July 30, 2014 
 *      Author: Sheng Lundquist
 */

#ifndef CUDADEVICE_HPP_
#define CUDADEVICE_HPP_


namespace PVCuda{
#include <stdio.h>
#include <cuda_runtime_api.h>
   
class CudaDevice {
protected:
   int device_id;                         // device id (normally 0 for GPU, 1 for CPU)

public:
   CudaDevice(int device);
   virtual ~CudaDevice();

   int initialize(int device);

   int id()  { return device_id; }

   //TODO do texture memory for readonly memory
   //CLBuffer * createBuffer(void* h_ptr);
   //CLBuffer * createBuffer(cl_mem_flags flags, size_t size, void * host_ptr);

   //CLBuffer * createReadBuffer(size_t size, void * host_ptr)
   //      { return createBuffer(CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, size, host_ptr); }
   //CLBuffer * createWriteBuffer(size_t size, void * host_ptr)
   //      { return createBuffer(CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, size, host_ptr); }
   //CLBuffer * createBuffer(size_t size, void * host_ptr)
   //      { return createBuffer(CL_MEM_COPY_HOST_PTR, size, host_ptr); }

   //CLKernel * createKernel(const char * filename, const char * name, const char * options);
   //CLKernel * createKernel(const char * filename, const char * name)
   //      { return createKernel(filename, name, NULL); }

   
//   int copyResultsBuffer(cl_mem output, void * results, size_t size);

   int query_device_info();
   void query_device(int id);
   int get_max_threads();
   int get_max_block_size_dimension(int dimension);
   int get_max_grid_size_dimension(int dimension);
   int get_warp_size();
   size_t get_local_mem();

protected:
   int num_devices;                  // number of computing devices
   struct cudaDeviceProp device_props;
};

} // namespace PV

#endif /* CLDEVICE_HPP_ */
