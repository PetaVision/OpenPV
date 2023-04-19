#ifndef FIRMTHRESHOLDCOSTFNLCAPROBELOCAL_HPP_
#define FIRMTHRESHOLDCOSTFNLCAPROBELOCAL_HPP_

#include "io/PVParams.hpp"
#include "probes/CostFunctionSum.hpp"
#include "probes/CostFunctions.hpp"
#include "probes/FirmThresholdCostFnProbeLocal.hpp"
#include "probes/NormProbeLocalTemplate.hpp"
#include <memory>

namespace PV {

class FirmThresholdCostFnLCAProbeLocal : public FirmThresholdCostFnProbeLocal {
  protected:
   virtual void ioParam_VThresh(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_VWidth(enum ParamsIOFlag ioFlag) override;

  public:
   FirmThresholdCostFnLCAProbeLocal(char const *objName, PVParams *params);
   virtual ~FirmThresholdCostFnLCAProbeLocal() {}

   void setFirmThresholdParams(double VThresh, double VWidth);

  protected:
   FirmThresholdCostFnLCAProbeLocal() {}
   void initialize(char const *objName, PVParams *params);
   void warnUnnecessaryParameter(char const *paramName);
};

} // namespace PV

#endif // FIRMTHRESHOLDCOSTFNLCAPROBELOCAL_HPP_
