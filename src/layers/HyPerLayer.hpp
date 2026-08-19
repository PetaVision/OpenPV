/**
 * HyPerLayer.hpp
 *
 *  Created on: Aug 3, 2008
 *      Author: dcoates
 *
 *  The top of the hierarchy for layer classes.
 *
 */

#ifndef HYPERLAYER_HPP_
#define HYPERLAYER_HPP_

#include "components/InternalStateBuffer.hpp"
#include "components/LayerInputBuffer.hpp"
#include "layers/BaseLayer.hpp"

namespace PV {

/**
 * The basic layer type implementing a LayerInput (GSyn) and a HyPerInternalStateBuffer (V).
 * The V buffer computes its values using the GSyn channels and the channelCoefficients from params.
 * The A buffer (which is extended) then copies V (which is restricted) to itself.
 */
class HyPerLayer : public BaseLayer {
  public:
   HyPerLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~HyPerLayer();

   float const *getV() const {
      return mActivityComponent->getComponentByType<InternalStateBuffer>()->getBufferData();
   }
   float *getV() {
      return mActivityComponent->getComponentByType<InternalStateBuffer>()->getReadWritePointer();
   }

  protected:
   HyPerLayer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual void fillComponentTable() override;
   virtual LayerInputBuffer *createLayerInput();
   virtual ActivityComponent *createActivityComponent() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

#ifdef PV_USE_CUDA
   Response::Status layerCopyFromGpu(std::shared_ptr<LayerCopyFromGpuMessage const> message) override;
#endif // PV_USE_CUDA

   // Data members
  protected:
   LayerInputBuffer *mLayerInput = nullptr;
};

} // namespace PV

#endif /* HYPERLAYER_HPP_ */
