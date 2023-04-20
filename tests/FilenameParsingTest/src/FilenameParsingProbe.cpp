/*
 * FilenameParsingProbe.cpp
 *
 *  Created on: April 20, 2017
 *      Author: peteschultz
 */

#include "FilenameParsingProbe.hpp"
#include <columns/BaseObject.hpp>
#include <columns/Communicator.hpp>
#include <columns/Messages.hpp>
#include <components/BasePublisherComponent.hpp>
#include <components/InputActivityBuffer.hpp>
#include <components/InputLayerNameParam.hpp>
#include <components/PhaseParam.hpp>
#include <include/PVLayerLoc.h>
#include <io/PVParams.hpp>
#include <layers/FilenameParsingLayer.hpp>
#include <observerpattern/BaseMessage.hpp>
#include <observerpattern/Response.hpp>
#include <probes/TargetLayerComponent.hpp>
#include <utils/PVAssert.hpp>
#include <utils/PVLog.hpp>
#include <utils/conversions.hpp>

#include <cmath>

FilenameParsingProbe::FilenameParsingProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

FilenameParsingProbe::~FilenameParsingProbe() {}

Response::Status FilenameParsingProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseObject::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   status = status + mProbeTargetLayerLocator->communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mFilenameParsingLayer =
         dynamic_cast<FilenameParsingLayer *>(mProbeTargetLayerLocator->getTargetLayer());
   FatalIf(
         mFilenameParsingLayer == nullptr,
         "%s requires the target layer to be a FilenameParsingLayer.\n",
         getDescription_c());

   auto *inputLayerNameParam  = mFilenameParsingLayer->getComponentByType<InputLayerNameParam>();
   char const *inputLayerName = inputLayerNameParam->getLinkedObjectName();
   pvAssert(inputLayerName);
   auto *inputBuffer = message->mObjectTable->findObject<InputActivityBuffer>(inputLayerName);
   FatalIf(
         inputBuffer == nullptr,
         "%s: %s inputLayerName \"%s\" does not link to an input layer.\n",
         getDescription_c(),
         mFilenameParsingLayer->getDescription_c(),
         inputLayerName);
   mInputDisplayPeriod = inputBuffer->getDisplayPeriod();
   return Response::SUCCESS;
}

void FilenameParsingProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   mProbeTargetLayerLocator = std::make_shared<TargetLayerComponent>(name, params);
   // createComponents() must be called before the base class's initialize(),
   // because BaseObject::initialize() calls the ioParamsFillGroup() method,
   // which calls each component's ioParamsFillGroup() method.
   BaseObject::initialize(name, params, comm);
}

int FilenameParsingProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseObject::ioParamsFillGroup(ioFlag);
   mProbeTargetLayerLocator->ioParamsFillGroup(ioFlag);
   return status;
}

void FilenameParsingProbe::initMessageActionMap() {
   BaseObject::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerOutputStateMessage const>(msgptr);
      return respondLayerOutputState(castMessage);
   };
   mMessageActionMap.emplace("LayerOutputState", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ProbeWriteParamsMessage const>(msgptr);
      return respondProbeWriteParams(castMessage);
   };
   mMessageActionMap.emplace("ProbeWriteParams", action);
}

PV::Response::Status
FilenameParsingProbe::outputState(std::shared_ptr<PV::LayerOutputStateMessage const> message) {
   double simTime = message->mTime;
   if (simTime == 0.0) {
      return Response::NO_ACTION;
   } // FilenameParsingLayer hasn't updated.

   double deltaTime         = message->mDeltaTime;
   double const displayTime = (simTime - deltaTime) / mInputDisplayPeriod;
   int const displayNumber  = (int)std::floor(displayTime);
   // From t=0 to the first display flip, displayNumber is 0.
   // From then until the second display flip, displayNumber is 1, etc.

   auto ioMPIBlock          = getCommunicator()->getIOMPIBlock();
   int mpiBatchIndex        = ioMPIBlock->getStartBatch() + ioMPIBlock->getBatchIndex();
   auto *publisherComponent = mFilenameParsingLayer->getComponentByType<BasePublisherComponent>();
   FatalIf(
         publisherComponent == nullptr,
         "Target layer \"%s\" does not have a BasePublisherComponent.\n",
         mFilenameParsingLayer->getName());
   PVLayerLoc const *loc      = publisherComponent->getLayerLoc();
   int const localBatchWidth  = loc->nbatch;
   int const globalBatchWidth = localBatchWidth * ioMPIBlock->getGlobalBatchDimension();
   int const localBatchStart  = mpiBatchIndex * localBatchWidth;
   int const numCategories    = (int)mCategories.size();
   int const numExtended      = publisherComponent->getNumExtended();
   int const nxExt            = loc->nx + loc->halo.lt + loc->halo.rt;
   int const nyExt            = loc->ny + loc->halo.dn + loc->halo.up;
   bool failed                = false;
   for (int b = 0; b < localBatchWidth; b++) {
      float const *activity = publisherComponent->getLayerData(0) + b * numExtended;
      int globalBatchIndex  = localBatchStart + b;
      int imageIndex        = (globalBatchIndex + displayNumber * globalBatchWidth) % numCategories;
      int expectedCategory  = mCategories[imageIndex];

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
   return Response::SUCCESS;
}

Response::Status FilenameParsingProbe::respondLayerOutputState(
      std::shared_ptr<LayerOutputStateMessage const> message) {
   auto status          = Response::SUCCESS;
   int targetLayerPhase = mFilenameParsingLayer->getComponentByType<PhaseParam>()->getPhase();
   if (message->mPhase == targetLayerPhase) {
      status = outputState(message);
   }
   return status;
}

Response::Status FilenameParsingProbe::respondProbeWriteParams(
      std::shared_ptr<ProbeWriteParamsMessage const> message) {
   writeParams();
   return Response::SUCCESS;
}
