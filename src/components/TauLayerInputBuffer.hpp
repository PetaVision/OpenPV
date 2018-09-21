/*
 * TauLayerInputBuffer.hpp
 *
 *  Created on: Sep 18, 2018
 *      Author: Pete Schultz
 */

#ifndef TAULAYERINPUTBUFFER_HPP_
#define TAULAYERINPUTBUFFER_HPP_

#include "components/LayerInputBuffer.hpp"
#include "include/default_params.h"

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class TauLayerInputBuffer : public LayerInputBuffer {
  protected:
   /**
    * List of parameters needed from the TauLayerInputBuffer class
    * @name HyPerLayer Parameters
    * @{
    */

   /**
    * @brief timeConstantTau: the time constant,
    * returned by getChannelTimeConstant for any channel
    */
   virtual void ioParam_timeConstantTau(enum ParamsIOFlag ioFlag);

   /** @} */
  public:
   TauLayerInputBuffer(char const *name, HyPerCol *hc);

   virtual ~TauLayerInputBuffer();

  protected:
   TauLayerInputBuffer() {}

   int initialize(char const *name, HyPerCol *hc);
   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual void initChannelTimeConstants() override;

  protected:
   double mTimeConstantTau = 1.0;
};

} // namespace PV

#endif // TAULAYERINPUTBUFFER_HPP_
