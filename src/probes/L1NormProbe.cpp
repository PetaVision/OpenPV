#include "L1NormProbe.hpp"
#include <memory>

namespace PV {

L1NormProbe::L1NormProbe(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

void L1NormProbe::createProbeLocal(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mProbeLocal = std::make_shared<L1NormProbeLocal>(params, defaults);
}

void L1NormProbe::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   AbstractNormProbe::initialize(params, defaults, comm);
}

} // namespace PV
