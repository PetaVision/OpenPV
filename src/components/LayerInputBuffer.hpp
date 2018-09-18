/*
 * LayerInputBuffer.hpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#ifndef LAYERINPUTBUFFER_HPP_
#define LAYERINPUTBUFFER_HPP_

#include "components/BufferComponent.hpp"
#include "components/LayerInputBuffer.hpp"
// #include "components/LayerReceiveComponent.hpp"
#include "initv/BaseInitV.hpp"

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class LayerInputBuffer : public BufferComponent {
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

   virtual void updateBuffer(double simTime, double deltaTime) override;

   float *getLayerInput() { return mBufferData.data(); }
   float *getLayerInput(int ch) { return &mBufferData.data()[ch * getBufferSizeAcrossBatch()]; }
   // TODO: remove. External access to mBufferData should be read-only, except through updateBuffer

  protected:
   LayerInputBuffer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

  protected:
   BaseInitV *mInitVObject = nullptr;
   char *mInitVTypeString  = nullptr;
   // GSyn component
};

} // namespace PV

#endif // LAYERINPUTBUFFER_HPP_
