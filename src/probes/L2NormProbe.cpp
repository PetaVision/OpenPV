#include "L2NormProbe.hpp"
#include "L2NormProbeAggregator.hpp"
#include <memory>

namespace PV {

L2NormProbe::L2NormProbe(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

void L2NormProbe::createProbeAggregator(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   mProbeAggregator =
         std::make_shared<L2NormProbeAggregator>(paramsIO, comm->getLocalMPIBlock());
}

void L2NormProbe::createProbeLocal(std::shared_ptr<ParamsIO> paramsIO) {
   mProbeLocal = std::make_shared<L2NormProbeLocal>(paramsIO);
}

void L2NormProbe::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   AbstractNormProbe::initialize(paramsIO, comm);
}

} // namespace PV
