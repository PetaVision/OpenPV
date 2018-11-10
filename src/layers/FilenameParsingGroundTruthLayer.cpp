/*
 * FilenameParsingGroundTruthLayer.cpp
 *
 *  Created on: Nov 10, 2014
 *      Author: wchavez
 */

#include "FilenameParsingGroundTruthLayer.hpp"

#include "components/ActivityComponentActivityOnly.hpp"
#include "components/FilenameParsingActivityBuffer.hpp"
#include "components/InputLayerNameParam.hpp"

namespace PV {
FilenameParsingGroundTruthLayer::FilenameParsingGroundTruthLayer(const char *name, HyPerCol *hc) {
   initialize(name, hc);
}

FilenameParsingGroundTruthLayer::~FilenameParsingGroundTruthLayer() {}

void FilenameParsingGroundTruthLayer::createComponentTable(char const *description) {
   HyPerLayer::createComponentTable(description);
   mInputLayerNameParam = createInputLayerNameParam();
   if (mInputLayerNameParam) {
      addUniqueComponent(mInputLayerNameParam->getDescription(), mInputLayerNameParam);
   }
}

LayerInputBuffer *FilenameParsingGroundTruthLayer::createLayerInput() { return nullptr; }

ActivityComponent *FilenameParsingGroundTruthLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<FilenameParsingActivityBuffer>(getName(), parent);
}

InputLayerNameParam *FilenameParsingGroundTruthLayer::createInputLayerNameParam() {
   return new InputLayerNameParam(getName(), parent);
}

Response::Status FilenameParsingGroundTruthLayer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerLayer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *inputLayerNameParam = getComponentByType<InputLayerNameParam>();
   pvAssert(inputLayerNameParam);
   if (!inputLayerNameParam->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   ComponentBasedObject *inputObject = nullptr;
   try {
      inputObject = inputLayerNameParam->findLinkedObject(message->mHierarchy);
   } catch (std::invalid_argument &e) {
      Fatal().printf("%s: %s\n", getDescription_c(), e.what());
   }
   pvAssert(inputObject);
   mInputLayer = dynamic_cast<InputLayer *>(inputObject);
   FatalIf(
         mInputLayer == nullptr,
         "%s inputLayerName \"%s\" points to an object that is not an InputLayer.\n",
         getDescription_c(),
         inputLayerNameParam->getLinkedObjectName());
   return Response::SUCCESS;
}

bool FilenameParsingGroundTruthLayer::needUpdate(double simTime, double dt) const {
   return mInputLayer->needUpdate(simTime, dt);
}

} // end namespace PV
