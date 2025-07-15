#include "ActivityBufferStatsProbeLocal.hpp"
#include "probes/BufferParamActivitySpecified.hpp"

namespace PV {

ActivityBufferStatsProbeLocal::ActivityBufferStatsProbeLocal(std::shared_ptr<ParamsIO> paramsIO) {
   initialize(paramsIO);
}

void ActivityBufferStatsProbeLocal::initialize(std::shared_ptr<ParamsIO> paramsIO) {
   StatsProbeLocal::initialize(paramsIO);
   setBufferParam<BufferParamActivitySpecified>(paramsIO);
}

} // namespace PV
