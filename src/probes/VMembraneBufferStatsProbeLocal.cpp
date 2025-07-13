#include "VMembraneBufferStatsProbeLocal.hpp"
#include "probes/BufferParamVMembraneSpecified.hpp"

namespace PV {

VMembraneBufferStatsProbeLocal::VMembraneBufferStatsProbeLocal(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   initialize(params, defaults);
}

void VMembraneBufferStatsProbeLocal::initialize(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   StatsProbeLocal::initialize(params, defaults);
   setBufferParam<BufferParamVMembraneSpecified>(params, defaults);
}

} // namespace PV
