#include "StatsProbeLocal.hpp"
#include "StatsProbeTypes.hpp"

#include "components/ActivityComponent.hpp"
#include "components/BasePublisherComponent.hpp"
#include "components/InternalStateBuffer.hpp"
#include "structures/PVLayerLoc.hpp"
#include "probes/BufferParamUserSpecified.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.hpp"

namespace PV {
StatsProbeLocal::StatsProbeLocal(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   initialize(params, defaults);
}

template <>
float const *
StatsProbeLocal::calculateBatchElementStart<StatsBufferType::A>(int localBatchIndex) const {
   PVLayerLoc const *loc = getLayerLoc();
   int nxExt             = loc->nx + loc->halo.lt + loc->halo.rt;
   int nyExt             = loc->ny + loc->halo.dn + loc->halo.up;
   int numExtended       = nxExt * nyExt * loc->nf;
   return findDataBufferA() + localBatchIndex * numExtended;
}

template <>
float const *
StatsProbeLocal::calculateBatchElementStart<StatsBufferType::V>(int localBatchIndex) const {
   PVLayerLoc const *loc = getLayerLoc();
   int numNeurons        = loc->nx * loc->ny * loc->nf;
   return findDataBufferV() + localBatchIndex * numNeurons;
}

void StatsProbeLocal::calculateStats(double simTime, ProbeData<LayerStats> &values) const {
   values.reset(simTime);
   int nbatch = getLayerLoc()->nbatch;
   pvAssert(nbatch == static_cast<int>(values.size()));
   for (int b = 0; b < nbatch; ++b) {
      LayerStats &elementStats = values.getValue(b);
      switch (getBufferType()) {
         case StatsBufferType::A: calculateValues<StatsBufferType::A>(elementStats, b); break;
         case StatsBufferType::V: calculateValues<StatsBufferType::V>(elementStats, b); break;
         default: Fatal().printf("Unrecognized StatsBufferType\n");
      }
   }
}

void StatsProbeLocal::clearStoredValues() { mStoredValues.clear(); }

float const *StatsProbeLocal::findDataBufferA() const {
   auto layerDataA = mTargetLayer->getComponentByType<BasePublisherComponent>();
   FatalIf(
         layerDataA == nullptr,
         "Probe %s target layer \"%s\" does not have an activity buffer.\n",
         getName_c(),
         mTargetLayer->getName());
   return layerDataA->getLayerData();
}

float const *StatsProbeLocal::findDataBufferV() const {
   auto *activityComponent = mTargetLayer->getComponentByType<ActivityComponent>();
   FatalIf(
         activityComponent == nullptr,
         "Probe %s target layer \"%s\": cannot find membrane potential.\n",
         getName_c(),
         mTargetLayer->getName());
   auto layerDataV = activityComponent->getComponentByType<InternalStateBuffer>();
   FatalIf(
         layerDataV == nullptr,
         "Probe %s target layer \"%s\" does not have a membrane potential.\n",
         getName_c(),
         mTargetLayer->getName());
   return layerDataV->getBufferData();
}

void StatsProbeLocal::initialize(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   ProbeComponent::initialize(params, defaults);
   setBufferParam<BufferParamUserSpecified>(params, defaults);
}

void StatsProbeLocal::initializeState(HyPerLayer *targetLayer) { mTargetLayer = targetLayer; }

void StatsProbeLocal::ioParam_buffer(ParamsIOSwitch ioSwitch) {
   mBufferParam->ioParam_buffer(ioSwitch);
}

void StatsProbeLocal::ioParam_nnzThreshold(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "nnzThreshold", &mNnzThreshold, false /*warnIfAbsentFlag*/);
}

void StatsProbeLocal::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_buffer(ioSwitch);
   ioParam_nnzThreshold(ioSwitch);
}

void StatsProbeLocal::storeValues(double simTime) {
   ProbeData<LayerStats> newValues(simTime, getLayerLoc()->nbatch);
   calculateStats(simTime, newValues);
   mStoredValues.store(newValues);
}

template <>
int StatsProbeLocal::calculateOffset<StatsBufferType::A>(int k) const {
   PVLayerLoc const *loc = getLayerLoc();
   int kExt              = kIndexExtended(
         k, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
   return kExt;
}

template <>
int StatsProbeLocal::calculateOffset<StatsBufferType::V>(int k) const {
   return k;
}

} // namespace PV
