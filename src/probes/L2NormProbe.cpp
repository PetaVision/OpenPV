#include "L2NormProbe.hpp"
#include "L2NormProbeAggregator.hpp"
#include <memory>

namespace PV {

L2NormProbe::L2NormProbe(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void L2NormProbe::createProbeAggregator(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mProbeAggregator =
         std::make_shared<L2NormProbeAggregator>(name, params, comm->getLocalMPIBlock());
}

void L2NormProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<L2NormProbeLocal>(name, params);
}

void L2NormProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   AbstractNormProbe::initialize(name, params, comm);
}

} // namespace PV
