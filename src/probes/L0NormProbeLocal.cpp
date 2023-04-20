#include "L0NormProbeLocal.hpp"
#include <memory>

namespace PV {

L0NormProbeLocal::L0NormProbeLocal(char const *objName, PVParams *params) {
   initialize(objName, params);
}

std::shared_ptr<L0CostFunctionSum const> L0NormProbeLocal::createCostFunctionSum() {
   auto costFunction = std::make_shared<L0CostFunction>(mNnzThreshold);
   auto norm         = std::make_shared<L0CostFunctionSum>(costFunction);
   return norm;
}

void L0NormProbeLocal::initialize(char const *objName, PVParams *params) {
   BaseL0NormProbeLocal::initialize(objName, params);
}

void L0NormProbeLocal::ioParam_nnzThreshold(enum ParamsIOFlag ioFlag) {
   getParams()->ioParamValue(ioFlag, getName_c(), "nnzThreshold", &mNnzThreshold, mNnzThreshold);
}

void L0NormProbeLocal::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   BaseL0NormProbeLocal::ioParamsFillGroup(ioFlag);
   ioParam_nnzThreshold(ioFlag);
}

} // namespace PV
