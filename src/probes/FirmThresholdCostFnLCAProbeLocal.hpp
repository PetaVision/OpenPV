#ifndef FIRMTHRESHOLDCOSTFNLCAPROBELOCAL_HPP_
#define FIRMTHRESHOLDCOSTFNLCAPROBELOCAL_HPP_

#include "probes/CostFunctionSum.hpp"
#include "probes/CostFunctions.hpp"
#include "probes/FirmThresholdCostFnProbeLocal.hpp"
#include "probes/NormProbeLocalTemplate.hpp"
#include <memory>

namespace PV {

class FirmThresholdCostFnLCAProbeLocal : public FirmThresholdCostFnProbeLocal {
  protected:
   virtual void ioParam_VThresh(ParamsIOSwitch ioSwitch) override;
   virtual void ioParam_VWidth(ParamsIOSwitch ioSwitch) override;

  public:
   FirmThresholdCostFnLCAProbeLocal(std::shared_ptr<ParamsIO> paramsIO);
   virtual ~FirmThresholdCostFnLCAProbeLocal() {}

   void setFirmThresholdParams(double VThresh, double VWidth);

  protected:
   FirmThresholdCostFnLCAProbeLocal() {}
   void initialize(std::shared_ptr<ParamsIO> paramsIO);
   void warnUnnecessaryParameter(char const *paramName);
};

} // namespace PV

#endif // FIRMTHRESHOLDCOSTFNLCAPROBELOCAL_HPP_
