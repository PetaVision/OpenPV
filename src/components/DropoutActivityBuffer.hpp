/*
 * DropoutActivityBuffer.hpp
 *
 *  Created on: Nov 7, 2016
 *      Author: Austin Thresher
 */

#ifndef DROPOUTACTIVITYBUFFER_HPP_
#define DROPOUTACTIVITYBUFFER_HPP_

#include "components/ANNActivityBuffer.hpp"

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class DropoutActivityBuffer : public ANNActivityBuffer {
  protected:
   /**
    * List of parameters used by the DropoutActivityBuffer class
    * @name DropoutLayer Parameters
    * @{
    */

   /**
    * @brief probability: The probability, as an integer from 0 to 99 inclusive,
    * giving percentage probability that a given activity neuron drops out.
    * Values above 99 are truncated to 99.
    *
    */
   virtual void ioParam_probability(enum ParamsIOFlag ioFlag);

   /** @} */
  public:
   DropoutActivityBuffer(char const *name, HyPerCol *hc);

   virtual ~DropoutActivityBuffer();

   bool usingVerticesListInParams() const { return mVerticesListInParams; }

  protected:
   DropoutActivityBuffer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   /**
    * Updates as ANNActivityBuffer updates, and then applies random dropout based on
    * the Probability parameter.
    */
   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   int mProbability = 0; // Value from 0-99 indicating per-neuron chance of dropout
};

} // namespace PV

#endif // DROPOUTACTIVITYBUFFER_HPP_
