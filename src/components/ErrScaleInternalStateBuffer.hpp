/*
 * ErrScaleInternalStateBuffer.hpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#ifndef ERRSCALEINTERNALSTATEBUFFER_HPP_
#define ERRSCALEINTERNALSTATEBUFFER_HPP_

#include "components/HyPerInternalStateBuffer.hpp"

namespace PV {

/**
 * A membrane potential component that reads the excitatory channel of a LayerInputBuffer and
 * computes V = GSynExc * GSynExc. Used by ANNSquaredLayer.
 */
class ErrScaleInternalStateBuffer : public HyPerInternalStateBuffer {
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
   ErrScaleInternalStateBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~ErrScaleInternalStateBuffer();

  protected:
   ErrScaleInternalStateBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   float mErrScale = 1.0f;
};

} // namespace PV

#endif // ERRSCALEINTERNALSTATEBUFFER_HPP_
