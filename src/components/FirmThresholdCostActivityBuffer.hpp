/*
 * FirmThresholdCostActivityBuffer.hpp
 *
 *  Created on: Apr 2, 2019
 *      Author: pschultz
 */

#ifndef FIRMTHRESHOLDCOSTACTIVITYBUFFER_HPP_
#define FIRMTHRESHOLDCOSTACTIVITYBUFFER_HPP_

#include "components/HyPerActivityBuffer.hpp"

namespace PV {

/**
 * A component to compute the activity fo a FirmThresholdCostLayer
 */
class FirmThresholdCostActivityBuffer : public HyPerActivityBuffer {
  protected:
   /**
    * List of parameters used by the FirmThresholdCostActivityBuffer class
    * @name FirmThresholdCostLayer Parameters
    * @{
    */

   /**
    * @brief VThresh:
    * The threshold value to use in calculating the cost function. This parameter is required.
    */
   virtual void ioParam_VThresh(enum ParamsIOFlag ioFlag);

   /**
    * @brief VWidth:
    * The width parameter to use in calculating the cost function. The default is zero.
    * linearly between A=AMin when V=VThresh and A=VThresh+VWidth-AShift when V=VThresh+VWidth.
    * Default is zero.
    */
   virtual void ioParam_VWidth(enum ParamsIOFlag ioFlag);

   /** @} */
  public:
   FirmThresholdCostActivityBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~FirmThresholdCostActivityBuffer();

   float getVThresh() const { return mVThresh; }
   float getVWidth() const { return mVWidth; }

  protected:
   FirmThresholdCostActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   float mVThresh = 0.0;
   float mVWidth  = 0.0f;
};

} // namespace PV

#endif // FIRMTHRESHOLDCOSTACTIVITYBUFFER_HPP_
