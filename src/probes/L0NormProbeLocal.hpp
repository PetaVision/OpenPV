#ifndef L0NORMPROBELOCAL_HPP_
#define L0NORMPROBELOCAL_HPP_

#include "probes/CostFunctionSum.hpp"
#include "probes/CostFunctions.hpp"
#include "probes/NormProbeLocalTemplate.hpp"
#include <memory>

namespace PV {

typedef CostFunctionSum<L0CostFunction> L0CostFunctionSum;
typedef NormProbeLocalTemplate<L0CostFunctionSum> BaseL0NormProbeLocal;

class L0NormProbeLocal : public BaseL0NormProbeLocal {
  protected:
   virtual void ioParam_nnzThreshold(ParamsIOSwitch ioSwitch);

  public:
   L0NormProbeLocal(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
   virtual ~L0NormProbeLocal() {}
   virtual void ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

  protected:
   L0NormProbeLocal() {}
   virtual std::shared_ptr<L0CostFunctionSum const> createCostFunctionSum() override;
   void initialize(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);

  protected:
   double mNnzThreshold = 0.0;
};

} // namespace PV

#endif // L0NORMPROBELOCAL_HPP_
