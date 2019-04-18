/*
 * FirmThresholdCostFnLCAProbe.hpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#ifndef FIRMTHRESHOLDCOSTFNLCAPROBE_HPP_
#define FIRMTHRESHOLDCOSTFNLCAPROBE_HPP_

#include "FirmThresholdCostFnProbe.hpp"

namespace PV {

/**
 * A special case of FirmThresholdCostFnProbe, to be used when the target layer
 * is an LCA layer
 * with a hard-threshold transfer function.  The corresponding cost function is
 * the norm
 * measured by FirmThresholdCostFnProbe, with coefficient Vth, where Vth is the
 * target LCA layer's
 * VThresh.
 */
class FirmThresholdCostFnLCAProbe : public FirmThresholdCostFnProbe {
  public:
   FirmThresholdCostFnLCAProbe(const char *name, HyPerCol *hc);
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual ~FirmThresholdCostFnLCAProbe() {}

  protected:
   FirmThresholdCostFnLCAProbe();
   int initFirmThresholdCostFnLCAProbe(const char *name, HyPerCol *hc) {
      return FirmThresholdCostFnProbe::initialize(name, hc);
   }

   /**
    * FirmThresholdCostFnLCAProbe does not read coefficient from params,
    * but computes it from VThresh of the target layer.
    */
   virtual void ioParam_coefficient(enum ParamsIOFlag ioFlag) override {
   } // coefficient is set from targetLayer during communicateInitInfo.

  private:
   int initialize_base() { return PV_SUCCESS; }
}; // class FirmThresholdCostFnLCAProbe

} /* namespace PV */

#endif /* FIRMTHRESHOLDCOSTFNLCAPROBE_HPP_ */
