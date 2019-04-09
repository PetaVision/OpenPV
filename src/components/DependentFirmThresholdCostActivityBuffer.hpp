/*
 * DependentFirmThresholdCostActivityBuffer.hpp
 *
 *  Created on: Apr 2, 2019
 *      Author: pschultz
 */

#ifndef DEPENDENTFIRMTHRESHOLDCOSTACTIVITYBUFFER_HPP_
#define DEPENDENTFIRMTHRESHOLDCOSTACTIVITYBUFFER_HPP_

#include "components/FirmThresholdCostActivityBuffer.hpp"

namespace PV {

/**
 * A component to compute the activity fo a FirmThresholdCostLayer
 */
class DependentFirmThresholdCostActivityBuffer : public FirmThresholdCostActivityBuffer {
  protected:
   /**
    * List of parameters used by the DependentFirmThresholdCostActivityBuffer class
    * @name FirmThresholdCostLayer Parameters
    * @{
    */

   /**
    * @brief VThresh:
    * The threshold value to use in calculating the cost function. This parameter is required.
    */
   virtual void ioParam_VThresh(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief VWidth:
    * The width parameter to use in calculating the cost function. The default is zero.
    * linearly between A=AMin when V=VThresh and A=VThresh+VWidth-AShift when V=VThresh+VWidth.
    * Default is zero.
    */
   virtual void ioParam_VWidth(enum ParamsIOFlag ioFlag) override;

   /** @} */
  public:
   DependentFirmThresholdCostActivityBuffer(
         char const *name,
         PVParams *params,
         Communicator const *comm);

   virtual ~DependentFirmThresholdCostActivityBuffer();

   float getVThresh() const { return mVThresh; }
   float getVWidth() const { return mVWidth; }

  protected:
   DependentFirmThresholdCostActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
};

} // namespace PV

#endif // DEPENDENTFIRMTHRESHOLDCOSTACTIVITYBUFFER_HPP_
