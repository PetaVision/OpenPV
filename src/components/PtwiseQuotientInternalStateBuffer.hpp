/*
 * PtwiseQuotientInternalStateBuffer.hpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#ifndef PTWISEQUOTIENTINTERNALSTATEBUFFER_HPP_
#define PTWISEQUOTIENTINTERNALSTATEBUFFER_HPP_

#include "components/InternalStateBuffer.hpp"

namespace PV {

/**
 * A membrane potential component that computes the pointwise quotient GSynExc / GSynInh.
 * Used by PtwiseQuotientLayer.
 */
class PtwiseQuotientInternalStateBuffer : public InternalStateBuffer {
  public:
   PtwiseQuotientInternalStateBuffer(char const *name, HyPerCol *hc);

   virtual ~PtwiseQuotientInternalStateBuffer();

   virtual void updateBuffer(double simTime, double deltaTime) override;

  protected:
   PtwiseQuotientInternalStateBuffer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

  protected:
};

} // namespace PV

#endif // PTWISEQUOTIENTINTERNALSTATEBUFFER_HPP_
