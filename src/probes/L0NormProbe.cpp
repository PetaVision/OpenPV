#include "L0NormProbe.hpp"
#include <memory>

namespace PV {

L0NormProbe::L0NormProbe(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

void L0NormProbe::createProbeLocal(std::shared_ptr<ParamsIO> paramsIO) {
   mProbeLocal = std::make_shared<L0NormProbeLocal>(paramsIO);
}

void L0NormProbe::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   AbstractNormProbe::initialize(paramsIO, comm);
}

} // namespace PV
