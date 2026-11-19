#include <algorithm> // std::min(), std::max()

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
StatsProbeLocal::StatsProbeLocal(char const *objName, PVParams *params) {
   initialize(objName, params);
}

void StatsProbeLocal::calculateActivityValues(LayerStats &stats, int localBatchIndex) const {
   PVLayerLoc const *loc = getLayerLoc();
   long numNeurons       = (long)loc->nx * (long)loc->ny * (long)loc->nf;
   double sum            = 0.0;
   double sumSquared     = 0.0;
   float min             = FLT_MAX;
   float max             = -FLT_MAX;
   long numNonzero       = 0;
   PVLayerCube cube      = findDataCube();
   if (cube.isSparse) {
      SparseList<float> const *sparseList                  = cube.activeIndices[localBatchIndex];
      std::vector<SparseEntry<float>> const &sparseEntries = sparseList->getContents();
      for (SparseEntry<float> const &e : sparseEntries) {
         float const a     = e.value;
         double const aDbl = static_cast<double>(a);
         sum += aDbl;
         sumSquared += aDbl * aDbl;
         numNonzero += (std::abs(a) > mNnzThreshold) ? 1 : 0;
         min = std::min(a, min);
         max = std::max(a, max);
      }
   }
   else {
      float const *data = cube.data[localBatchIndex];
      for (long k = 0; k < numNeurons; k++) {
         long kExt = kIndexExtended(
               k,
               loc->nx,
               loc->ny,
               loc->nf,
               loc->halo.lt,
               loc->halo.rt,
               loc->halo.dn,
               loc->halo.up);
         float const a     = data[kExt];
         double const aDbl = static_cast<double>(a);
         sum += aDbl;
         sumSquared += aDbl * aDbl;
         numNonzero += (std::abs(a) > mNnzThreshold) ? 1 : 0;
         min = std::min(a, min);
         max = std::max(a, max);
      }
   }
   stats.mSum        = sum;
   stats.mSumSquared = sumSquared;
   stats.mMin        = min;
   stats.mMax        = max;
   stats.mNumNeurons = numNeurons;
   stats.mNumNonzero = numNonzero;
}

void StatsProbeLocal::calculateInternalStateValues(LayerStats &stats, int localBatchIndex) const {
   PVLayerLoc const *loc = getLayerLoc();
   long numNeurons       = (long)loc->nx * (long)loc->ny * (long)loc->nf;
   float const *data     = &findVBuffer()[localBatchIndex * numNeurons];
   double sum            = 0.0;
   double sumSquared     = 0.0;
   float min             = FLT_MAX;
   float max             = -FLT_MAX;
   long numNonzero       = 0;
   for (long k = 0; k < numNeurons; k++) {
      float const a     = data[k];
      double const aDbl = static_cast<double>(a);
      sum += aDbl;
      sumSquared += aDbl * aDbl;
      numNonzero += (std::abs(a) > mNnzThreshold) ? 1 : 0;
      min = (a < min) ? a : min;
      max = (a > max) ? a : max;
   }
   stats.mSum        = sum;
   stats.mSumSquared = sumSquared;
   stats.mMin        = min;
   stats.mMax        = max;
   stats.mNumNeurons = numNeurons;
   stats.mNumNonzero = numNonzero;
}

void StatsProbeLocal::calculateStats(double simTime, ProbeData<LayerStats> &values) const {
   values.reset(simTime);
   int nbatch = getLayerLoc()->nbatch;
   pvAssert(nbatch == static_cast<int>(values.size()));
   for (int b = 0; b < nbatch; ++b) {
      LayerStats &elementStats = values.getValue(b);
      switch (getBufferType()) {
         case StatsBufferType::A: calculateActivityValues(elementStats, b); break;
         case StatsBufferType::V: calculateInternalStateValues(elementStats, b); break;
         default: Fatal().printf("Unrecognized StatsBufferType\n");
      }
   }
}

void StatsProbeLocal::clearStoredValues() { mStoredValues.clear(); }

PVLayerCube StatsProbeLocal::findDataCube() const {
   auto layerDataA = mTargetLayer->getComponentByType<BasePublisherComponent>();
   FatalIf(
         layerDataA == nullptr,
         "Probe %s target layer \"%s\" does not have an activity buffer.\n",
         getName_c(),
         mTargetLayer->getName());
   return layerDataA->getLayerCube();
}

float const *StatsProbeLocal::findVBuffer() const {
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

void StatsProbeLocal::initialize(char const *objName, PVParams *params) {
   ProbeComponent::initialize(objName, params);
   setBufferParam<BufferParamUserSpecified>(objName, params);
}

void StatsProbeLocal::initializeState(HyPerLayer *targetLayer) { mTargetLayer = targetLayer; }

void StatsProbeLocal::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   mBufferParam->ioParam_buffer(ioFlag);
}

void StatsProbeLocal::ioParam_nnzThreshold(enum ParamsIOFlag ioFlag) {
   bool warnIfAbsent         = true;
   float defaultNnzThreshold = 0.0f;
   getParams()->ioParamValue(
         ioFlag, getName_c(), "nnzThreshold", &mNnzThreshold, defaultNnzThreshold, warnIfAbsent);
}

void StatsProbeLocal::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_buffer(ioFlag);
   ioParam_nnzThreshold(ioFlag);
}

void StatsProbeLocal::storeValues(double simTime) {
   ProbeData<LayerStats> newValues(simTime, getLayerLoc()->nbatch);
   calculateStats(simTime, newValues);
   mStoredValues.store(newValues);
}

} // namespace PV
