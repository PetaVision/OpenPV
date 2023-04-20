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
 * A special case of FirmThresholdCostFnProbe, to be used when the target layer is an LCA layer
 * with a hard-threshold transfer function.  The corresponding cost function is the norm measured
 * by FirmThresholdCostFnProbe, with coefficient Vth, where Vth is the target LCA layer's VThresh.
 */
class FirmThresholdCostFnLCAProbe : public FirmThresholdCostFnProbe {
  public:
   FirmThresholdCostFnLCAProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~FirmThresholdCostFnLCAProbe() {}

  protected:
   FirmThresholdCostFnLCAProbe() {}

   virtual Response::Status allocateDataStructures() override;

   virtual void createProbeLocal(char const *name, PVParams *params) override;
   virtual void createEnergyProbeComponent(char const *name, PVParams *params) override;

   void initialize(const char *name, PVParams *params, Communicator const *comm);
}; // class FirmThresholdCostFnLCAProbe

} /* namespace PV */

#endif /* FIRMTHRESHOLDCOSTFNLCAPROBE_HPP_ */
