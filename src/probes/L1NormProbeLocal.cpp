#include "L1NormProbeLocal.hpp"
#include <memory>

namespace PV {

L1NormProbeLocal::L1NormProbeLocal(std::shared_ptr<ParamsIO> paramsIO) {
   initialize(paramsIO);
}

std::shared_ptr<CostFunctionSum<L1CostFunction> const> L1NormProbeLocal::createCostFunctionSum() {
   auto costFunction = std::make_shared<L1CostFunction>();
   auto norm         = std::make_shared<CostFunctionSum<L1CostFunction>>(costFunction);
   return norm;
}

void L1NormProbeLocal::initialize(std::shared_ptr<ParamsIO> paramsIO) {
   NormProbeLocalTemplate<CostFunctionSum<L1CostFunction>>::initialize(paramsIO);
}

} // namespace PV
