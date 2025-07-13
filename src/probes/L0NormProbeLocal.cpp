#include "L0NormProbeLocal.hpp"
#include <memory>

namespace PV {

L0NormProbeLocal::L0NormProbeLocal(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   initialize(params, defaults);
}

std::shared_ptr<L0CostFunctionSum const> L0NormProbeLocal::createCostFunctionSum() {
   auto costFunction = std::make_shared<L0CostFunction>(mNnzThreshold);
   auto norm         = std::make_shared<L0CostFunctionSum>(costFunction);
   return norm;
}

void L0NormProbeLocal::initialize(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   BaseL0NormProbeLocal::initialize(params, defaults);
}

void L0NormProbeLocal::ioParam_nnzThreshold(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "nnzThreshold", &mNnzThreshold);
}

void L0NormProbeLocal::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   BaseL0NormProbeLocal::ioParamsFillGroup(ioSwitch);
   ioParam_nnzThreshold(ioSwitch);
}

} // namespace PV
