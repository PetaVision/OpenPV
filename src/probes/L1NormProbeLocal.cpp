#include "L1NormProbeLocal.hpp"
#include <memory>

namespace PV {

L1NormProbeLocal::L1NormProbeLocal(char const *objName, PVParams *params) {
   initialize(objName, params);
}

std::shared_ptr<CostFunctionSum<L1CostFunction> const> L1NormProbeLocal::createCostFunctionSum() {
   auto costFunction = std::make_shared<L1CostFunction>();
   auto norm         = std::make_shared<CostFunctionSum<L1CostFunction>>(costFunction);
   return norm;
}

void L1NormProbeLocal::initialize(char const *objName, PVParams *params) {
   NormProbeLocalTemplate<CostFunctionSum<L1CostFunction>>::initialize(objName, params);
}

} // namespace PV
