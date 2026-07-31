/*
 * ShrunkenPatchTestProbe.cpp
 *
 *  Created on:
 *      Author: garkenyon
 */

#include "ShrunkenPatchTestProbe.hpp"
#include <columns/Communicator.hpp>
#include <components/BasePublisherComponent.hpp>
#include <components/LayerGeometry.hpp>
#include <structures/PVLayerLoc.hpp>
#include <io/PVParams.hpp>
#include <layers/HyPerLayer.hpp>
#include <probes/ActivityBufferStatsProbeLocal.hpp>
#include <probes/ProbeData.hpp>
#include <probes/StatsProbeAggregator.hpp>
#include <probes/StatsProbeImmediate.hpp>
#include <probes/StatsProbeTypes.hpp>
#include <utils/PVLog.hpp>
#include <utils/conversions.hpp>

#include <cmath>
#include <cstdlib>
#include <memory>

namespace PV {

/**
 * @filename
 * @type
 * @msg
 */
ShrunkenPatchTestProbe::ShrunkenPatchTestProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm)
      : StatsProbeImmediate() {
   initialize(name, params, comm);
}

void ShrunkenPatchTestProbe::checkStats() {
   HyPerLayer *l         = getTargetLayer();
   const PVLayerLoc *loc = l->getLayerLoc();
   long numNeurons       = l->getNumNeurons();

   // NxpShrunken must be an integer multiple of the layer's nxScale, and nxScale must be a positive
   // integral power of 2.
   // The correct values of the layer activity is a function of its column index, that depends on
   // NxpShrunken.
   // If NxpShrunken is an odd multiple of nxScale, the patch is not really shrunken, and then the
   // correct values
   // of the layer activity are [<0.5> <0.5> <1.5> <1.5> <2.5> <2.5> ...], where angle brackets
   // indicate that the given
   // value is repeated nxScale/2 times.
   //
   // If NxpShrunken is an even multiple, the correct values of the layer activity are
   // [ <0.0> <1.0> <1.0> <2.0> <2.0> ...]
   //
   // This assumes the connection with l as the post-synaptic layer has a pre-synaptic layer with
   // nxScale=1.
   // There isn't a convenient way for a ShrunkenPatchTestProbe object to ensure that that's the
   // case.

   if (mCorrectValues == NULL) {
      int nx         = loc->nx;
      mCorrectValues = (float *)malloc((size_t)nx * sizeof(float));

      int xScaleLog2 = getTargetLayer()->getComponentByType<LayerGeometry>()->getXScale();

      if (xScaleLog2 >= 0) {
         Fatal().printf(
               "%s: layer \"%s\" must have nxScale > 1.\n", getDescription_c(), l->getName());
      }
      int cellSize = (int)std::nearbyint(std::pow(2, -xScaleLog2));
      int kx0       = (loc->kx0) / cellSize;
      FatalIf(kx0 * cellSize != loc->kx0, "Test failed.\n");
      int halfCellSize = cellSize / 2;
      FatalIf(halfCellSize * 2 != cellSize, "Test failed.\n");
      int numHalfCells = nx / halfCellSize;
      FatalIf(numHalfCells * halfCellSize != nx, "Test failed.\n");
      int cellsInPatch = mNxpShrunken / cellSize;
      if (mNxpShrunken != cellsInPatch * cellSize) {
         Fatal().printf(
               "ShrunkenPatchTestProbe \"%s\" error: NxpShrunken must be an integer multiple of "
               "layer \"%s\" nxScale=%d.\n",
               getName(),
               l->getName(),
               cellSize);
      }
      int nxpSizeParity = cellsInPatch % 2;

      int idx = 0;
      for (int hc = 0; hc < numHalfCells; hc++) {
         int m               = 2 * ((hc + 1 - nxpSizeParity) / 2) + nxpSizeParity;
         float correctValue = (float)kx0 + 0.5f * (float)m;
         for (int k = 0; k < halfCellSize; k++) {
            mCorrectValues[idx++] = correctValue;
         }
      }
      FatalIf(!(idx == nx), "Test failed.\n");
   }
   FatalIf(!(mCorrectValues != NULL), "Test failed.\n");

   float tol = 1e-4f;

   float const *buf =
         getTargetLayer()->getComponentByType<BasePublisherComponent>()->getLayerData();

   auto const &storedValues           = mProbeAggregator->getStoredValues();
   auto numTimestamps                 = storedValues.size();
   int lastTimestampIndex             = static_cast<int>(numTimestamps) - 1;
   ProbeData<LayerStats> const &stats = storedValues.getData(lastTimestampIndex);

   double simTime = stats.getTimestamp();
   if (simTime >= 3.0) {
      for (long k = 0; k < numNeurons; k++) {
         long kex = kIndexExtended(
               k,
               loc->nx,
               loc->ny,
               loc->nf,
               loc->halo.lt,
               loc->halo.rt,
               loc->halo.dn,
               loc->halo.up);
         int x = kxPos(k, loc->nx, loc->ny, loc->nf);
         if (std::fabs(buf[kex] - mCorrectValues[x]) > tol) {
            int y = kyPos(k, loc->nx, loc->ny, loc->nf);
            int f = featureIndex(k, loc->nx, loc->ny, loc->nf);
            Fatal().printf(
                  "%s: Incorrect value %f (should be %f) in process %d, x=%d, y=%d, f=%d\n",
                  l->getDescription_c(),
                  (double)buf[kex],
                  (double)mCorrectValues[x],
                  mCommunicator->globalCommRank(),
                  x,
                  y,
                  f);
         }
      }
   }
}

void ShrunkenPatchTestProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(name, params);
}

void ShrunkenPatchTestProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   mCorrectValues = NULL;
   StatsProbeImmediate::initialize(name, params, comm);
}

int ShrunkenPatchTestProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbeImmediate::ioParamsFillGroup(ioFlag);
   ioParam_nxpShrunken(ioFlag);
   ioParam_nypShrunken(ioFlag);
   return status;
}

void ShrunkenPatchTestProbe::ioParam_nxpShrunken(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, getName(), "nxpShrunken", &mNxpShrunken);
   return;
}

void ShrunkenPatchTestProbe::ioParam_nypShrunken(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, getName(), "nypShrunken", &mNypShrunken);
   return;
}

ShrunkenPatchTestProbe::~ShrunkenPatchTestProbe() { free(mCorrectValues); }
} // namespace PV
