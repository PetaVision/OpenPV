#include "L1NormLCAProbeLocal.hpp"

namespace PV {

L1NormLCAProbeLocal::L1NormLCAProbeLocal(std::shared_ptr<ParamsIO> paramsIO) {
   initialize(paramsIO);
}

void L1NormLCAProbeLocal::initialize(std::shared_ptr<ParamsIO> paramsIO) {
   L1NormProbeLocal::initialize(paramsIO);
}

} // namespace PV
