/*
 * CudaDevice.cu
 *
 *  Created on: Aug 5, 2014
 *      Author: Sheng Lundquist
 */

#include "CudaDevice.hpp"
#include "cuda_util.hpp"

#ifdef PV_USE_CUDNN
#include <cudnn.h>
#endif

namespace PVCuda {

CudaDevice::CudaDevice(int device) {
   mDeviceID = device;
   initialize(mDeviceID);
   // Set amount of device memory to global memory
   mDeviceMem = mDeviceProps.totalGlobalMem;
}

CudaDevice::~CudaDevice() {
   handleError(cudaStreamDestroy(mStream), "Cuda Device Destructor");

#ifdef PV_USE_CUDNN
   if (mHandle) {
      cudnnDestroy((cudnnHandle_t)mHandle);
   }
#endif
}

void CudaDevice::incrementConvKernels() { mNumConvKernels++; }

long CudaDevice::reserveMem(size_t size) {
   mDeviceMem -= size;
   return mDeviceMem;
}

int CudaDevice::getNumDevices() {
   int returnVal;
   handleError(cudaGetDeviceCount(&returnVal), "Static getting device count");
   return returnVal;
}

int CudaDevice::initialize(int device) {
   int status = 0;

#ifdef PV_USE_CUDA
   handleError(cudaDeviceReset(), "Device resetting in initialize");

   handleError(cudaGetDeviceCount(&mNumDevices), "Getting device count");
   FatalIf(mNumDevices == 0, "No CUDA device found.\n");

   handleError(cudaSetDevice(device), "Setting device");

   handleError(cudaStreamCreate(&mStream), "Creating stream");

   handleError(cudaGetDeviceProperties(&mDeviceProps, device), "Getting device properties");

   status = 0;
#ifdef PV_USE_CUDNN
   // Testing cudnn here
   cudnnHandle_t tmpHandle;
   cudnnStatus_t cudnnStatus = cudnnCreate(&tmpHandle);
   if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
      Fatal(cudnnCreateError);
      switch (cudnnStatus) {
         case CUDNN_STATUS_NOT_INITIALIZED:
            cudnnCreateError.printf("cuDNN Runtime API initialization failed\n");
            break;
         case CUDNN_STATUS_ALLOC_FAILED:
            cudnnCreateError.printf("cuDNN resources could not be allocated\n");
            break;
         default:
            cudnnCreateError.printf("cudnnCreate error: %s\n", cudnnGetErrorString(cudnnStatus));
            break;
      }
      exit(EXIT_FAILURE);
   }
   cudnnStatus = cudnnSetStream(tmpHandle, mStream);
   if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
      Fatal().printf("cudnnSetStream error: %s\n", cudnnGetErrorString(cudnnStatus));
   }

   mHandle = (void *)tmpHandle;
#endif // PV_USE_CUDNN
#endif // PV_USE_CUDA

   return status;
}

void CudaDevice::syncDevice() { handleError(cudaDeviceSynchronize(), "Synchronizing device"); }

int CudaDevice::query_device_info() {
   // query and print information about the devices found
   //
   InfoLog().printf("\n");
   InfoLog().printf("Number of Cuda devices found: %d\n", mNumDevices);
   InfoLog().printf("\n");

   for (int i = 0; i < mNumDevices; i++) {
      query_device(i);
   }
   return 0;
}

CudaBuffer *CudaDevice::createBuffer(size_t size, std::string const *str) {
   long memLeft = reserveMem(size);
   InfoLog() << "Reserving " << size << " bytes of VRAM";
   if (str) {
      InfoLog() << " (" << *str << ")";
   }
   InfoLog() << ". " << mDeviceMem << " bytes remaining.\n";
   if (memLeft < 0) {
      InfoLog().flush();
      Fatal().printf("CudaDevice createBuffer: out of memory\n");
   }
   return (new CudaBuffer(size, &mDeviceProps, mStream));
}

void CudaDevice::query_device(int id) {
   struct cudaDeviceProp props;
   // Use own props if current device
   if (id == mDeviceID) {
      props = mDeviceProps;
   }
   // Otherwise, generate props
   else {
      handleError(cudaGetDeviceProperties(&props, id), "Getting device properties");
   }
   int clockRateInKHz;
   cudaDeviceGetAttribute(&clockRateInKHz, cudaDevAttrClockRate, id);

   InfoLog().printf("device: %d\n", id);
   InfoLog().printf("CUDA Device # %d == %s\n", id, props.name);

   InfoLog().printf("with %d units/cores", props.multiProcessorCount);

   InfoLog().printf(" at %f MHz\n", static_cast<double>(clockRateInKHz) * 0.001);

   InfoLog().printf("\tMaximum threads group size == %d\n", props.maxThreadsPerBlock);

   InfoLog().printf("\tMaximum grid sizes == (");
   for (unsigned int i = 0; i < 3; i++)
      InfoLog().printf(" %d", props.maxGridSize[i]);
   InfoLog().printf(" )\n");

   InfoLog().printf("\tMaximum threads sizes == (");
   for (unsigned int i = 0; i < 3; i++)
      InfoLog().printf(" %d", props.maxThreadsDim[i]);
   InfoLog().printf(" )\n");

   InfoLog().printf("\tLocal mem size == %zu\n", props.sharedMemPerBlock);

   InfoLog().printf("\tGlobal mem size == %zu\n", props.totalGlobalMem);

   InfoLog().printf("\tConst mem size == %zu\n", props.totalConstMem);

   InfoLog().printf("\tRegisters per block == %d\n", props.regsPerBlock);

   InfoLog().printf("\tWarp size == %d\n", props.warpSize);

   InfoLog().printf("\n");
}

int CudaDevice::get_max_threads() { return mDeviceProps.maxThreadsPerBlock; }

int CudaDevice::get_max_block_size_dimension(int dimension) {
   if (dimension < 0 || dimension >= 3)
      return 0;
   return mDeviceProps.maxThreadsDim[dimension];
}

int CudaDevice::get_max_grid_size_dimension(int dimension) {
   if (dimension < 0 || dimension >= 3)
      return 0;
   return mDeviceProps.maxGridSize[dimension];
}

int CudaDevice::get_warp_size() { return mDeviceProps.warpSize; }

size_t CudaDevice::get_local_mem() { return mDeviceProps.sharedMemPerBlock; }
}
