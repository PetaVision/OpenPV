#include "L2NormProbe.hpp"
#include "L2NormProbeAggregator.hpp"
#include <memory>

namespace PV {

L2NormProbe::L2NormProbe(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

void L2NormProbe::createProbeAggregator(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   mProbeAggregator =
         std::make_shared<L2NormProbeAggregator>(params, defaults, comm->getLocalMPIBlock());
}

void L2NormProbe::createProbeLocal(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mProbeLocal = std::make_shared<L2NormProbeLocal>(params, defaults);
}

void L2NormProbe::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   AbstractNormProbe::initialize(params, defaults, comm);
}

} // namespace PV
