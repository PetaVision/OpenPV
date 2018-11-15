/*
 * CPTestInputInternalStateBuffer.hpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#ifndef CPTESTINPUTINTERNALSTATEBUFFER_HPP_
#define CPTESTINPUTINTERNALSTATEBUFFER_HPP_

#include "components/HyPerInternalStateBuffer.hpp"

namespace PV {

/**
 * A membrane potential component that initializes each neuron to its global index,
 * and then increases each neuron by one each time updateBuffer is called.
 * Used by several checkpoint system tests.
 */
class CPTestInputInternalStateBuffer : public HyPerInternalStateBuffer {
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
