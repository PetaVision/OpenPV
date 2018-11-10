/*
 * LayerInputBuffer.hpp
 *
 *  Created on: Sep 13, 2018 from the original HyPerLayer
 *      Author: Pete Schultz
 */

#ifndef LAYERINPUTBUFFER_HPP_
#define LAYERINPUTBUFFER_HPP_

#include "components/ComponentBuffer.hpp"
#include "delivery/LayerInputDelivery.hpp"

#ifdef PV_USE_CUDA
#include "arch/cuda/CudaTimer.hpp"
#endif // PV_USE_CUDA

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class LayerInputBuffer : public ComponentBuffer {
  protected:
   /**
    * List of parameters needed from the LayerInputBuffer class
    * @name HyPerLayer Parameters
    * @{
    */

   /** @} */
  public:
   LayerInputBuffer(char const *name, HyPerCol *hc);

   virtual ~LayerInputBuffer();

   virtual void requireChannel(int channelNeeded);

   double getChannelTimeConstant(int channelCode) { return mChannelTimeConstants[channelCode]; }

   /**
    * Adds the given delivery object to the vector of delivery sources to receive input from.
    * The delivery object's post-synaptic layer should be the layer for which this member function
    * is called; however, to avoid circular dependencies among classes this requirement
    * is not checked.
    */
   void addDeliverySource(LayerInputDelivery *delivery);

   bool getHasReceived() const { return mHasReceived; }

  protected:
   LayerInputBuffer() {}

   int initialize(char const *name, HyPerCol *hc);
   virtual void initMessageActionMap() override;
   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   virtual void initChannelTimeConstants();

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   Response::Status
   respondLayerClearProgressFlags(std::shared_ptr<LayerClearProgressFlagsMessage const> message);

   Response::Status
   respondLayerRecvSynapticInput(std::shared_ptr<LayerRecvSynapticInputMessage const> message);

#ifdef PV_USE_CUDA
   Response::Status respondLayerCopyFromGpu(std::shared_ptr<LayerCopyFromGpuMessage const> message);
#endif // PV_USE_CUDA

   /**
    * Returns true if each layer that delivers input to this layer
    * has finished its MPI exchange for its delay; false if any of
    * them has not.
    */
   bool isAllInputReady();

   virtual void resetGSynBuffers(double simTime, double dt);

   /**
    * Calls deliver for each DeliverySource connecting to this buffer.
    */
   virtual void recvAllSynapticInput(double simTime, double deltaTime);

   virtual void updateBufferCPU(double simTime, double deltaTime) override;
#ifdef PV_USE_CUDA
   virtual void updateBufferGPU(double simTime, double deltaTime) override;
#endif // PV_USE_CUDA

  protected:
   std::vector<double> mChannelTimeConstants;
   std::vector<LayerInputDelivery *> mDeliverySources;
   bool mHasReceived = false;

   Timer *mReceiveInputTimer = nullptr;
#ifdef PV_USE_CUDA
   std::vector<LayerInputDelivery *> mGPUDeliverySources; // delivery sources that set recvGpu
   PVCuda::CudaTimer *mReceiveInputCudaTimer = nullptr;
#endif
};

} // namespace PV

#endif // LAYERINPUTBUFFER_HPP_
