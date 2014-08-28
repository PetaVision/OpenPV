/*
 * CudaDevice.cu
 *
 *  Created on: Aug 5, 2014
 *      Author: Sheng Lundquist
 */

#include "../../include/pv_arch.h"
#include "cuda_util.hpp"
#include "CudaDevice.hpp"
#include "CudaBuffer.hpp"

namespace PVCuda{

CudaDevice::CudaDevice(int device)
{
   this->device_id = device;
   initialize(device_id);
}

CudaDevice::~CudaDevice()
{
}

int CudaDevice::initialize(int device)
{
   int status = 0;

#ifdef PV_USE_CUDA
   handleError(cudaThreadExit());

   handleError(cudaGetDeviceCount(&num_devices));
   printf("Num devices: %d\n", num_devices);
   // get number of devices available
   //

   printf("Using device %d\n", device);
   handleError(cudaSetDevice(device));

   status = 0;
#endif // PV_USE_OPENCL

   return status;
}

int CudaDevice::query_device_info()
{
   // query and print information about the devices found
   //
   printf("\n");
   printf("Number of Cuda devices found: %d\n", num_devices);
   printf("\n");

   for (unsigned int i = 0; i < num_devices; i++) {
      query_device(i);
   }
   return 0;
}

void CudaDevice::query_device(int id)
{
   printf("device: %d\n", id);
   struct cudaDeviceProp props;
   handleError(cudaGetDeviceProperties(&props, id));

   if(id == device_id){
      this->device_props = props;
   }

   printf("CUDA Device # %d == %s\n", id, props.name);

   printf("with %d units/cores", props.multiProcessorCount);

   printf(" at %f MHz\n", (float)props.clockRate * .001);

   printf("\tMaximum threads group size == %d\n", props.maxThreadsPerBlock);
   
   printf("\tMaximum grid sizes == (");
   for (unsigned int i = 0; i < 3; i++) printf(" %d", props.maxGridSize[i]);
   printf(" )\n");

   printf("\tMaximum threads sizes == (");
   for (unsigned int i = 0; i < 3; i++) printf(" %d", props.maxThreadsDim[i]);
   printf(" )\n");

   printf("\tLocal mem size == %zu\n", props.sharedMemPerBlock);

   printf("\tGlobal mem size == %zu\n", props.totalGlobalMem);

   printf("\tConst mem size == %zu\n", props.totalConstMem);

   printf("\tRegisters per block == %d\n", props.regsPerBlock);

   printf("\tWarp size == %d\n", props.warpSize);

   printf("\n");
}

int CudaDevice::get_max_threads(){
   return device_props.maxThreadsPerBlock;
}

int CudaDevice::get_max_block_size_dimension(int dimension){
   if(dimension < 0 || dimension >= 3) return 0;
   return device_props.maxThreadsDim[dimension];
}

int CudaDevice::get_max_grid_size_dimension(int dimension){
   if(dimension < 0 || dimension >= 3) return 0;
   return device_props.maxGridSize[dimension];
}

int CudaDevice::get_warp_size(){
   return device_props.warpSize;
}

size_t CudaDevice::get_local_mem(){
   return device_props.sharedMemPerBlock;
}

}
