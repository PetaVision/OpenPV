#include "L0NormProbe.hpp"
#include <memory>

namespace PV {

L0NormProbe::L0NormProbe(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

void L0NormProbe::createProbeLocal(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mProbeLocal = std::make_shared<L0NormProbeLocal>(params, defaults);
}

void L0NormProbe::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   AbstractNormProbe::initialize(params, defaults, comm);
}

} // namespace PV
