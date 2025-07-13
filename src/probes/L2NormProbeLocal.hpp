#ifndef L2NORMPROBELOCAL_HPP_
#define L2NORMPROBELOCAL_HPP_

#include "probes/CostFunctionSum.hpp"
#include "probes/CostFunctions.hpp"
#include "probes/NormProbeLocalTemplate.hpp"
#include <memory>

namespace PV {

class L2NormProbeLocal : public NormProbeLocalTemplate<CostFunctionSum<L2CostFunction>> {
  public:
   L2NormProbeLocal(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
   virtual ~L2NormProbeLocal() {}

  protected:
   L2NormProbeLocal() {}
   virtual std::shared_ptr<CostFunctionSum<L2CostFunction> const> createCostFunctionSum() override;
   void initialize(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
};

} // namespace PV

#endif // L2NORMPROBELOCAL_HPP_
