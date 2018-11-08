/*
 * HyPerInternalStateBuffer.hpp
 *
 *  Created on: Oct 12, 2018 from the original HyPerLayer
 *      Author: Pete Schultz
 */

#ifndef HYPERINTERNALSTATEBUFFER_HPP_
#define HYPERINTERNALSTATEBUFFER_HPP_

#include "components/GSynInternalStateBuffer.hpp"
#include "components/LayerInputBuffer.hpp"

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class HyPerInternalStateBuffer : public GSynInternalStateBuffer {
  public:
   HyPerInternalStateBuffer(char const *name, PVParams *params, Communicator *comm);

   virtual ~HyPerInternalStateBuffer();

  protected:
   HyPerInternalStateBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual void requireInputChannels() override;

   /**
    * Computes the buffer as excitatory input minus inhibitory input from the LayerInput buffer.
    * The previous internal state has no effect on the new internal state.
    */
   virtual void updateBufferCPU(double simTime, double deltaTime) override;
};

} // namespace PV

#endif // HYPERINTERNALSTATEBUFFER_HPP_
