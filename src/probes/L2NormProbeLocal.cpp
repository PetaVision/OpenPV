#include "L2NormProbeLocal.hpp"
#include <memory>

namespace PV {

L2NormProbeLocal::L2NormProbeLocal(char const *objName, PVParams *params) {
   initialize(objName, params);
}

std::shared_ptr<CostFunctionSum<L2CostFunction> const> L2NormProbeLocal::createCostFunctionSum() {
   auto costFunction = std::make_shared<L2CostFunction>();
   auto norm         = std::make_shared<CostFunctionSum<L2CostFunction>>(costFunction);
   return norm;
}

void L2NormProbeLocal::initialize(char const *objName, PVParams *params) {
   NormProbeLocalTemplate<CostFunctionSum<L2CostFunction>>::initialize(objName, params);
}

} // namespace PV
