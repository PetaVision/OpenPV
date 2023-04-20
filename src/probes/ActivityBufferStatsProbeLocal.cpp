#include "ActivityBufferStatsProbeLocal.hpp"
#include "probes/BufferParamActivitySpecified.hpp"

namespace PV {

ActivityBufferStatsProbeLocal::ActivityBufferStatsProbeLocal(
      char const *objName,
      PVParams *params) {
   initialize(objName, params);
}

void ActivityBufferStatsProbeLocal::initialize(char const *objName, PVParams *params) {
   StatsProbeLocal::initialize(objName, params);
   setBufferParam<BufferParamActivitySpecified>(objName, params);
}

} // namespace PV
