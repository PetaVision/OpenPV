#ifndef FIRMTHRESHOLDCOSTFNPROBELOCAL_HPP_
#define FIRMTHRESHOLDCOSTFNPROBELOCAL_HPP_

#include "io/PVParams.hpp"
#include "probes/CostFunctionSum.hpp"
#include "probes/CostFunctions.hpp"
#include "probes/NormProbeLocalTemplate.hpp"
#include <memory>

namespace PV {

typedef CostFunctionSum<FirmThresholdCostFunction> FirmThresholdCostFunctionSum;
typedef NormProbeLocalTemplate<FirmThresholdCostFunctionSum> BaseFirmThresholdCostFnProbeLocal;

class FirmThresholdCostFnProbeLocal : public BaseFirmThresholdCostFnProbeLocal {
  protected:
   virtual void ioParam_VThresh(enum ParamsIOFlag ioFlag);
   virtual void ioParam_VWidth(enum ParamsIOFlag ioFlag);

  public:
   FirmThresholdCostFnProbeLocal(char const *objName, PVParams *params);
   virtual ~FirmThresholdCostFnProbeLocal() {}
   virtual void ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   FirmThresholdCostFnProbeLocal() {}
   virtual std::shared_ptr<FirmThresholdCostFunctionSum const> createCostFunctionSum() override;
   void initialize(char const *objName, PVParams *params);

  protected:
   double mVThresh = 0.0;
   double mVWidth  = 0.0;
};

} // namespace PV

#endif // FIRMTHRESHOLDCOSTFNPROBELOCAL_HPP_
