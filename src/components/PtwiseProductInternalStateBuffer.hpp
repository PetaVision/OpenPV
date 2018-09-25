/*
 * PtwiseProductInternalStateBuffer.hpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#ifndef PTWISEPRODUCTINTERNALSTATEBUFFER_HPP_
#define PTWISEPRODUCTINTERNALSTATEBUFFER_HPP_

#include "components/InternalStateBuffer.hpp"

namespace PV {

/**
 * A membrane potential component that computes the pointwise product GSynExc * GSynInh.
 * Used by PtwiseProductLayer.
 */
class PtwiseProductInternalStateBuffer : public InternalStateBuffer {
  public:
   PtwiseProductInternalStateBuffer(char const *name, HyPerCol *hc);

   virtual ~PtwiseProductInternalStateBuffer();

   virtual void updateBuffer(double simTime, double deltaTime) override;

  protected:
   PtwiseProductInternalStateBuffer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

  protected:
};

} // namespace PV

#endif // PTWISEPRODUCTINTERNALSTATEBUFFER_HPP_
