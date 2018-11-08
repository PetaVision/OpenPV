/*
 * CPTestInputInternalStateBuffer.hpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#ifndef CPTESTINPUTINTERNALSTATEBUFFER_HPP_
#define CPTESTINPUTINTERNALSTATEBUFFER_HPP_

#include "components/GSynInternalStateBuffer.hpp"

namespace PV {

/**
 * A membrane potential component that reads the excitatory channel of a LayerInputBuffer and
 * computes V = GSynExc * GSynExc. Used by ANNSquaredLayer.
 */
class CPTestInputInternalStateBuffer : public GSynInternalStateBuffer {
  public:
   CPTestInputInternalStateBuffer(char const *name, PVParams *params, Communicator *comm);

   virtual ~CPTestInputInternalStateBuffer();

  protected:
   CPTestInputInternalStateBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   Response::Status initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;
};

} // namespace PV

#endif // CPTESTINPUTINTERNALSTATEBUFFER_HPP_
