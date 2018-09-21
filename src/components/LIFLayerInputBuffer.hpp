/*
 * LIFLayerInputBuffer.hpp
 *
 *  Created on: Sep 18, 2018
 *      Author: Pete Schultz
 */

#ifndef LIFLAYERINPUTBUFFER_HPP_
#define LIFLAYERINPUTBUFFER_HPP_

#include "components/LayerInputBuffer.hpp"
#include "include/default_params.h"

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class LIFLayerInputBuffer : public LayerInputBuffer {
  protected:
   /**
    * List of parameters needed from the LIFLayerInputBuffer class
    * @name HyPerLayer Parameters
    * @{
    */

   /** @brief tauE: the time constant for the excitatory channel. */
   virtual void ioParam_tauE(enum ParamsIOFlag ioFlag);

   /** @brief tauI: the time constant for the inhibitory channel. */
   virtual void ioParam_tauI(enum ParamsIOFlag ioFlag);

   /** @brief tauIB: the time constant for the after-hyperpolarization. */
   virtual void ioParam_tauIB(enum ParamsIOFlag ioFlag);

   /** @} */
  public:
   LIFLayerInputBuffer(char const *name, HyPerCol *hc);

   virtual ~LIFLayerInputBuffer();

   void recvUnitInput(float *recvBuffer);

  protected:
   LIFLayerInputBuffer() {}

   int initialize(char const *name, HyPerCol *hc);
   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual void initChannelTimeConstants() override;

  protected:
   double mTauE  = (float)TAU_EXC;
   double mTauI  = (float)TAU_INH;
   double mTauIB = (float)TAU_INHB;
};

} // namespace PV

#endif // LIFLAYERINPUTBUFFER_HPP_
