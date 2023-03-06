#include "L0NormProbe.hpp"
#include <memory>

namespace PV {

L0NormProbe::L0NormProbe(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void L0NormProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<L0NormProbeLocal>(name, params);
}

void L0NormProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   AbstractNormProbe::initialize(name, params, comm);
}

} // namespace PV
