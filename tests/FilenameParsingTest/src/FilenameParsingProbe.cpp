/*
 * FilenameParsingProbe.cpp
 *
 *  Created on: April 20, 2017
 *      Author: peteschultz
 */

#include "FilenameParsingProbe.hpp"
#include <components/InputActivityBuffer.hpp>

FilenameParsingProbe::FilenameParsingProbe() { initialize_base(); }

/**
 * @filename
 */
FilenameParsingProbe::FilenameParsingProbe(
      const char *name,
      PV::PVParams *params,
      PV::Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
}

FilenameParsingProbe::~FilenameParsingProbe() {}

int FilenameParsingProbe::initialize_base() { return PV_SUCCESS; }

void FilenameParsingProbe::initialize(
      const char *name,
      PV::PVParams *params,
      PV::Communicator *comm) {
   LayerProbe::initialize(name, params, comm);
}

PV::Response::Status FilenameParsingProbe::communicateInitInfo(
      std::shared_ptr<PV::CommunicateInitInfoMessage const> message) {
   auto status = PV::LayerProbe::communicateInitInfo(message);
   if (!PV::Response::completed(status)) {
      return status;
   }

   char const *inputLayerName = parameters()->stringValue(getTargetName(), "inputLayerName", false);
   pvAssert(inputLayerName);
   std::string inputLayerString(inputLayerName);
   PV::InputLayer *inputLayer = message->mHierarchy->lookupByName<PV::InputLayer>(inputLayerString);
   pvAssert(inputLayer);
   auto *activityComponent = inputLayer->getComponentByType<PV::ActivityComponent>();
   pvAssert(activityComponent);
   auto *inputBuffer = activityComponent->getComponentByType<PV::InputActivityBuffer>();
   pvAssert(inputBuffer);
   mInputDisplayPeriod = inputBuffer->getDisplayPeriod();
   return PV::Response::SUCCESS;
}

PV::Response::Status FilenameParsingProbe::outputState(double simTime, double deltaTime) {
   if (simTime == 0.0) {
      return PV::Response::NO_ACTION;
   } // FilenameParsingGroundTruthLayer hasn't updated.

   double const displayTime = (simTime - deltaTime) / mInputDisplayPeriod;
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
   bool failed                = false;
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
            failed = true;
         }
      }
   }
   FatalIf(failed, "FilenameParsingProbe failed at t=%f\n", simTime);
   return PV::Response::SUCCESS;
}
