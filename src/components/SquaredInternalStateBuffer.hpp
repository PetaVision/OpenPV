/*
 * SquaredInternalStateBuffer.hpp
 *
 *  Created on: Sep 21, 2011
 *      Author: kpeterson
 */

#ifndef SQUAREDINTERNALSTATEBUFFER_HPP_
#define SQUAREDINTERNALSTATEBUFFER_HPP_

#include "components/GSynInternalStateBuffer.hpp"

namespace PV {

/**
 * A membrane potential component that reads the excitatory channel of a LayerInputBuffer and
 * computes V = GSynExc * GSynExc. Used by ANNSquaredLayer.
 */
class SquaredInternalStateBuffer : public GSynInternalStateBuffer {
  public:
   SquaredInternalStateBuffer(char const *name, PVParams *params, Communicator *comm);

   virtual ~SquaredInternalStateBuffer();

  protected:
   SquaredInternalStateBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual void requireInputChannels() override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;
};

} // namespace PV

#endif // SQUAREDINTERNALSTATEBUFFER_HPP_
