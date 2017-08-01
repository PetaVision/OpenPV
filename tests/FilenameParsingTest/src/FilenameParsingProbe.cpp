/*
 * FilenameParsingProbe.cpp
 *
 *  Created on: April 20, 2017
 *      Author: peteschultz
 */

#include "FilenameParsingProbe.hpp"

FilenameParsingProbe::FilenameParsingProbe() { initialize_base(); }

/**
 * @filename
 */
FilenameParsingProbe::FilenameParsingProbe(const char *name, PV::HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

FilenameParsingProbe::~FilenameParsingProbe() {}

int FilenameParsingProbe::initialize_base() { return PV_SUCCESS; }

int FilenameParsingProbe::initialize(const char *name, PV::HyPerCol *hc) {
   int status = LayerProbe::initialize(name, hc);
   return status;
}

int FilenameParsingProbe::communicateInitInfo(
      std::shared_ptr<PV::CommunicateInitInfoMessage const> message) {
   int status = PV::LayerProbe::communicateInitInfo(message);
   if (status != PV_SUCCESS) {
      return status;
   }

   char const *inputLayerName =
         parent->parameters()->stringValue(getTargetName(), "inputLayerName", false);
   pvAssert(inputLayerName);
   PV::InputLayer *inputLayer = message->lookup<PV::InputLayer>(std::string(inputLayerName));
   pvAssert(inputLayer);
   mInputDisplayPeriod = inputLayer->getDisplayPeriod();
   return PV_SUCCESS;
}

int FilenameParsingProbe::outputState(double timestamp) {
   int status = PV_SUCCESS;
   if (timestamp == 0.0) {
      return status;
   } // FilenameParsingGroundTruthLayer hasn't updated.

   double const displayTime = (timestamp - parent->getDeltaTime()) / mInputDisplayPeriod;
   int const displayNumber  = (int)std::floor(displayTime);
   // From t=0 to the first display flip, displayNumber is 0.
   // From then until the second display flip, displayNumber is 1, etc.

   int mpiBatchIndex          = getMPIBlock()->getStartBatch() + getMPIBlock()->getBatchIndex();
   PVLayerLoc const *loc      = getTargetLayer()->getLayerLoc();
   int const localBatchWidth  = loc->nbatch;
   int const globalBatchWidth = localBatchWidth * getMPIBlock()->getGlobalBatchDimension();
   int const localBatchStart  = mpiBatchIndex * localBatchWidth;
   int const numCategories    = (int)mCategories.size();
   int const numExtended      = getTargetLayer()->getNumExtended();
   int const nxExt            = loc->nx + loc->halo.lt + loc->halo.rt;
   int const nyExt            = loc->ny + loc->halo.dn + loc->halo.up;
   for (int b = 0; b < localBatchWidth; b++) {
      float const *activity =
            getTargetLayer()->getLayerData(0) + b * getTargetLayer()->getNumExtended();
      int globalBatchIndex = localBatchStart + b;
      int imageIndex       = (globalBatchIndex + displayNumber * globalBatchWidth);
      imageIndex %= (int)mCategories.size();
      int expectedCategory = mCategories[imageIndex];

      for (int k = 0; k < numExtended; k++) {
         int f               = featureIndex(k, nxExt, nyExt, loc->nf);
         float expectedValue = f == expectedCategory ? 1.0f : 0.0f;
         float observedValue = activity[k];
         if (expectedValue != observedValue) {
            status = PV_FAILURE;
         }
      }
   }
   FatalIf(status != PV_SUCCESS, "FilenameParsingProbe failed at t=%f\n", timestamp);
   return status;
}
