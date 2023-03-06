#include "L1NormProbe.hpp"
#include <memory>

namespace PV {

L1NormProbe::L1NormProbe(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void L1NormProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<L1NormProbeLocal>(name, params);
}

void L1NormProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   AbstractNormProbe::initialize(name, params, comm);
}

} // namespace PV
