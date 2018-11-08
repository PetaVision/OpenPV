/*
 * PtwiseQuotientInternalStateBuffer.hpp
 *
 * created by gkenyon, 06/2016g
 * based on PtwiseProductLayer Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#ifndef PTWISEQUOTIENTINTERNALSTATEBUFFER_HPP_
#define PTWISEQUOTIENTINTERNALSTATEBUFFER_HPP_

#include "components/GSynInternalStateBuffer.hpp"

namespace PV {

/**
 * A membrane potential component that computes the pointwise quotient GSynExc / GSynInh.
 * Used by PtwiseQuotientLayer.
 */
class PtwiseQuotientInternalStateBuffer : public GSynInternalStateBuffer {
  public:
   PtwiseQuotientInternalStateBuffer(char const *name, PVParams *params, Communicator *comm);

   virtual ~PtwiseQuotientInternalStateBuffer();

  protected:
   PtwiseQuotientInternalStateBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual void requireInputChannels() override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
};

} // namespace PV

#endif // PTWISEQUOTIENTINTERNALSTATEBUFFER_HPP_
