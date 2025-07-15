#include "VMembraneBufferStatsProbeLocal.hpp"
#include "probes/BufferParamVMembraneSpecified.hpp"

namespace PV {

VMembraneBufferStatsProbeLocal::VMembraneBufferStatsProbeLocal(std::shared_ptr<ParamsIO> paramsIO) {
   initialize(paramsIO);
}

void VMembraneBufferStatsProbeLocal::initialize(std::shared_ptr<ParamsIO> paramsIO) {
   StatsProbeLocal::initialize(paramsIO);
   setBufferParam<BufferParamVMembraneSpecified>(paramsIO);
}

} // namespace PV
