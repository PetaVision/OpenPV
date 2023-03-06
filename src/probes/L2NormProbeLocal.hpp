#ifndef L2NORMPROBELOCAL_HPP_
#define L2NORMPROBELOCAL_HPP_

#include "io/PVParams.hpp"
#include "probes/CostFunctionSum.hpp"
#include "probes/CostFunctions.hpp"
#include "probes/NormProbeLocalTemplate.hpp"
#include <memory>

namespace PV {

class L2NormProbeLocal : public NormProbeLocalTemplate<CostFunctionSum<L2CostFunction>> {
  public:
   L2NormProbeLocal(char const *objName, PVParams *params);
   virtual ~L2NormProbeLocal() {}

  protected:
   L2NormProbeLocal() {}
   virtual std::shared_ptr<CostFunctionSum<L2CostFunction> const> createCostFunctionSum() override;
   void initialize(char const *objName, PVParams *params);
};

} // namespace PV

#endif // L2NORMPROBELOCAL_HPP_
