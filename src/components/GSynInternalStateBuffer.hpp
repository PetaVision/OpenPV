/*
 * GSynInternalStateBuffer.hpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#ifndef GSYNINTERNALSTATEBUFFER_HPP_
#define GSYNINTERNALSTATEBUFFER_HPP_

#include "components/InternalStateBuffer.hpp"
#include "components/LayerInputBuffer.hpp"

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer,
 * with a pointer to a LayerInputBuffer to serve as a GSyn.
 */
class GSynInternalStateBuffer : public InternalStateBuffer {
  public:
   GSynInternalStateBuffer(char const *name, HyPerCol *hc);

   virtual ~GSynInternalStateBuffer();

  protected:
   GSynInternalStateBuffer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /**
    * A virtual function, called by communicateInitInfo, intended for derived classes to call
    * mLayerInput->requireChannel with whatever channels the class's updateBuffer uses.
    */
   virtual void requireInputChannels();

  protected:
   LayerInputBuffer *mLayerInput = nullptr;
};

} // namespace PV

#endif // GSYNINTERNALSTATEBUFFER_HPP_
