#include "L1NormLCAProbeLocal.hpp"

namespace PV {

L1NormLCAProbeLocal::L1NormLCAProbeLocal(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults) {
   initialize(params, defaults);
}

void L1NormLCAProbeLocal::initialize(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   L1NormProbeLocal::initialize(params, defaults);
}

} // namespace PV
