/*
 * PtwiseProductInternalStateBuffer.hpp
 *
 *  Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#ifndef PTWISEPRODUCTINTERNALSTATEBUFFER_HPP_
#define PTWISEPRODUCTINTERNALSTATEBUFFER_HPP_

#include "components/GSynInternalStateBuffer.hpp"

namespace PV {

/**
 * A membrane potential component that computes the pointwise product GSynExc * GSynInh.
 * Used by PtwiseProductLayer.
 */
class PtwiseProductInternalStateBuffer : public GSynInternalStateBuffer {
  public:
   PtwiseProductInternalStateBuffer(char const *name, HyPerCol *hc);

   virtual ~PtwiseProductInternalStateBuffer();

  protected:
   PtwiseProductInternalStateBuffer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual void requireInputChannels() override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
};

} // namespace PV

#endif // PTWISEPRODUCTINTERNALSTATEBUFFER_HPP_
