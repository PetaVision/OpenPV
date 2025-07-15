#include "L1NormProbe.hpp"
#include <memory>

namespace PV {

L1NormProbe::L1NormProbe(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

void L1NormProbe::createProbeLocal(std::shared_ptr<ParamsIO> paramsIO) {
   mProbeLocal = std::make_shared<L1NormProbeLocal>(paramsIO);
}

void L1NormProbe::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   AbstractNormProbe::initialize(paramsIO, comm);
}

} // namespace PV
