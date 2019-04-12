/*
 * PresynapticPerspectiveGPUDelivery.hpp
 *
 *  Created on: Jan 10, 2018
 *      Author: Pete Schultz
 */

#ifndef PRESYNAPTICPERSPECTIVEGPUDELIVERY_HPP_
#define PRESYNAPTICPERSPECTIVEGPUDELIVERY_HPP_

#include "HyPerDelivery.hpp"
#include "arch/cuda/CudaBuffer.hpp"
#include "cudakernels/CudaRecvPre.hpp"

namespace PV {

/**
 * The delivery class for HyPerConns using the presynaptic perspective on the CPU,
 * with accumulate type "convolve".
 */
class PresynapticPerspectiveGPUDelivery : public HyPerDelivery {
  public:
   PresynapticPerspectiveGPUDelivery(char const *name, PVParams *params, Communicator const *comm);

   virtual ~PresynapticPerspectiveGPUDelivery();

   /**
    * The method that delivers presynaptic activity to the given postsynaptic channel.
    * It loops over presynaptic neurons, skipping over any whose activity is zero
    * (to take advantage of sparsity). Each neuron then modifies the region of the post channel
    * that the weights argument specifies for that pre-synaptic neuron.
    */
   virtual void deliver(float *destBuffer) override;

   virtual void deliverUnitInput(float *recvBuffer) override;

  protected:
   PresynapticPerspectiveGPUDelivery();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status
   setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   virtual Response::Status copyInitialStateToGPU() override;

   void initializeRecvKernelArgs();

   // Data members
  protected:
   PVCuda::CudaRecvPre *mRecvKernel          = nullptr;
   PVCuda::CudaBuffer *mDevicePatches        = nullptr;
   PVCuda::CudaBuffer *mDeviceGSynPatchStart = nullptr;

}; // end class PresynapticPerspectiveGPUDelivery

} // end namespace PV

#endif // PRESYNAPTICPERSPECTIVEGPUDELIVERY_HPP_
