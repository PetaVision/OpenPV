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
  protected:
   /**
    * List of parameters needed from the PresynapticPerspectiveGPUDelivery class
    * @name PresynapticPerspectiveGPUDelivery Parameters
    * @{
    */

   /**
    * @brief receiveGpu: PresynapticPerspectiveGPUDelivery always sets receiveGpu to true.
    * The receiveGpu=false case is handled by the PresynapticPerspectiveConvolveDelivery
    * and PresynapticPerspectiveStochasticDelivery classes.
    */
   virtual void ioParam_receiveGpu(enum ParamsIOFlag ioFlag) override;
   /** @} */ // End of list of BaseDelivery parameters.

  public:
   PresynapticPerspectiveGPUDelivery(char const *name, HyPerCol *hc);

   virtual ~PresynapticPerspectiveGPUDelivery();

   /**
    * The method that delivers presynaptic activity to the given postsynaptic channel.
    * It loops over presynaptic neurons, skipping over any whose activity is zero
    * (to take advantage of sparsity). Each neuron then modifies the region of the post channel
    * that the weights argument specifies for that pre-synaptic neuron.
    */
   virtual void deliver() override;

   virtual void deliverUnitInput(float *recvBuffer) override;

  protected:
   PresynapticPerspectiveGPUDelivery();

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status
   setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   void initializeRecvKernelArgs();

   void allocateThreadGSyn();

   // Data members
  protected:
   std::vector<std::vector<float>> mThreadGSyn; // needed since deliverUnitInput is not on the GPU
   PVCuda::CudaRecvPre *mRecvKernel   = nullptr;
   PVCuda::CudaBuffer *mDevicePatches = nullptr;
   PVCuda::CudaBuffer *mDeviceGSynPatchStart = nullptr;

}; // end class PresynapticPerspectiveGPUDelivery

} // end namespace PV

#endif // PRESYNAPTICPERSPECTIVEGPUDELIVERY_HPP_
