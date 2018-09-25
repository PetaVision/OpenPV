/*
 * SquaredInternalStateBuffer.hpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#ifndef SQUAREDINTERNALSTATEBUFFER_HPP_
#define SQUAREDINTERNALSTATEBUFFER_HPP_

#include "components/InternalStateBuffer.hpp"

namespace PV {

/**
 * A membrane potential component that reads the excitatory channel of a LayerInputBuffer and
 * computes V = GSynExc * GSynExc. Used by ANNSquaredLayer.
 */
class SquaredInternalStateBuffer : public InternalStateBuffer {
  public:
   SquaredInternalStateBuffer(char const *name, HyPerCol *hc);

   virtual ~SquaredInternalStateBuffer();

   virtual void updateBuffer(double simTime, double deltaTime) override;

  protected:
   SquaredInternalStateBuffer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

  private:
  protected:
};

} // namespace PV

#endif // SQUAREDINTERNALSTATEBUFFER_HPP_
