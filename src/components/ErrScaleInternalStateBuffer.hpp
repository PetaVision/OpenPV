/*
 * ErrScaleInternalStateBuffer.hpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#ifndef ERRSCALEINTERNALSTATEBUFFER_HPP_
#define ERRSCALEINTERNALSTATEBUFFER_HPP_

#include "components/InternalStateBuffer.hpp"

namespace PV {

/**
 * A membrane potential component that reads the excitatory channel of a LayerInputBuffer and
 * computes V = GSynExc * GSynExc. Used by ANNSquaredLayer.
 */
class ErrScaleInternalStateBuffer : public InternalStateBuffer {
  protected:
   /**
    * List of parameters used by the ANNErrorLayer class
    * @name ANNErrorLayer Parameters
    * @{
    */

   /**
    * @brief: errScale: The input to the error layer is multiplied by errScale before applying the
    * threshold.
    */
   virtual void ioParam_errScale(enum ParamsIOFlag ioFlag);

  public:
   ErrScaleInternalStateBuffer(char const *name, HyPerCol *hc);

   virtual ~ErrScaleInternalStateBuffer();

   virtual void updateBuffer(double simTime, double deltaTime) override;

  protected:
   ErrScaleInternalStateBuffer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   float mErrScale = 1.0f;
};

} // namespace PV

#endif // ERRSCALEINTERNALSTATEBUFFER_HPP_
