#include "VMembraneBufferStatsProbeLocal.hpp"
#include "probes/BufferParamVMembraneSpecified.hpp"

namespace PV {

VMembraneBufferStatsProbeLocal::VMembraneBufferStatsProbeLocal(
      char const *objName,
      PVParams *params) {
   initialize(objName, params);
}

void VMembraneBufferStatsProbeLocal::initialize(char const *objName, PVParams *params) {
   StatsProbeLocal::initialize(objName, params);
   setBufferParam<BufferParamVMembraneSpecified>(objName, params);
}

} // namespace PV
