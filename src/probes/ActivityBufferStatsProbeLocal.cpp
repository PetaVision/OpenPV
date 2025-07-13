#include "ActivityBufferStatsProbeLocal.hpp"
#include "probes/BufferParamActivitySpecified.hpp"

namespace PV {

ActivityBufferStatsProbeLocal::ActivityBufferStatsProbeLocal(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   initialize(params, defaults);
}

void ActivityBufferStatsProbeLocal::initialize(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   StatsProbeLocal::initialize(params, defaults);
   setBufferParam<BufferParamActivitySpecified>(params, defaults);
}

} // namespace PV
