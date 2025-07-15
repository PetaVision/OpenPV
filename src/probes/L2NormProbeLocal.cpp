#include "L2NormProbeLocal.hpp"
#include <memory>

namespace PV {

L2NormProbeLocal::L2NormProbeLocal(std::shared_ptr<ParamsIO> paramsIO) {
   initialize(paramsIO);
}

std::shared_ptr<CostFunctionSum<L2CostFunction> const> L2NormProbeLocal::createCostFunctionSum() {
   auto costFunction = std::make_shared<L2CostFunction>();
   auto norm         = std::make_shared<CostFunctionSum<L2CostFunction>>(costFunction);
   return norm;
}

void L2NormProbeLocal::initialize(std::shared_ptr<ParamsIO> paramsIO) {
   NormProbeLocalTemplate<CostFunctionSum<L2CostFunction>>::initialize(paramsIO);
}

} // namespace PV
