#ifndef FIRMTHRESHOLDCOSTFNPROBELOCAL_HPP_
#define FIRMTHRESHOLDCOSTFNPROBELOCAL_HPP_

#include "probes/CostFunctionSum.hpp"
#include "probes/CostFunctions.hpp"
#include "probes/NormProbeLocalTemplate.hpp"
#include <memory>

namespace PV {

typedef CostFunctionSum<FirmThresholdCostFunction> FirmThresholdCostFunctionSum;
typedef NormProbeLocalTemplate<FirmThresholdCostFunctionSum> BaseFirmThresholdCostFnProbeLocal;

class FirmThresholdCostFnProbeLocal : public BaseFirmThresholdCostFnProbeLocal {
  protected:
   virtual void ioParam_VThresh(ParamsIOSwitch ioSwitch);
   virtual void ioParam_VWidth(ParamsIOSwitch ioSwitch);

  public:
   FirmThresholdCostFnProbeLocal(std::shared_ptr<ParamsIO> paramsIO);
   virtual ~FirmThresholdCostFnProbeLocal() {}
   virtual void ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

  protected:
   FirmThresholdCostFnProbeLocal() {}
   virtual std::shared_ptr<FirmThresholdCostFunctionSum const> createCostFunctionSum() override;
   void initialize(std::shared_ptr<ParamsIO> paramsIO);

  protected:
   double mVThresh = 0.0;
   double mVWidth  = 0.0;
};

} // namespace PV

#endif // FIRMTHRESHOLDCOSTFNPROBELOCAL_HPP_
