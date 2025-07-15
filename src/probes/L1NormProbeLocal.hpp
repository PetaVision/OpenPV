#ifndef L1NORMPROBELOCAL_HPP_
#define L1NORMPROBELOCAL_HPP_

#include "probes/CostFunctionSum.hpp"
#include "probes/CostFunctions.hpp"
#include "probes/NormProbeLocalTemplate.hpp"
#include <memory>

namespace PV {

class L1NormProbeLocal : public NormProbeLocalTemplate<CostFunctionSum<L1CostFunction>> {
  public:
   L1NormProbeLocal(std::shared_ptr<ParamsIO> paramsIO);
   virtual ~L1NormProbeLocal() {}

  protected:
   L1NormProbeLocal() {}
   virtual std::shared_ptr<CostFunctionSum<L1CostFunction> const> createCostFunctionSum() override;
   void initialize(std::shared_ptr<ParamsIO> paramsIO);
};

} // namespace PV

#endif // L1NORMPROBELOCAL_HPP_
