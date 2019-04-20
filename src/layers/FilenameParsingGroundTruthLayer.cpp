/*
 * FilenameParsingGroundTruthLayer.cpp
 *
 *  Created on: Nov 10, 2014
 *      Author: wchavez
 */

#include "FilenameParsingGroundTruthLayer.hpp"

#include "components/ActivityComponentActivityOnly.hpp"
#include "components/FilenameParsingActivityBuffer.hpp"
#include "components/FilenameParsingLayerUpdateController.hpp"

namespace PV {
FilenameParsingGroundTruthLayer::FilenameParsingGroundTruthLayer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

FilenameParsingGroundTruthLayer::~FilenameParsingGroundTruthLayer() {}

void FilenameParsingGroundTruthLayer::fillComponentTable() {
   HyPerLayer::fillComponentTable();
   mInputLayerNameParam = createInputLayerNameParam();
   if (mInputLayerNameParam) {
      addUniqueComponent(mInputLayerNameParam);
   }
}

LayerInputBuffer *FilenameParsingGroundTruthLayer::createLayerInput() { return nullptr; }

LayerUpdateController *FilenameParsingGroundTruthLayer::createLayerUpdateController() {
   return new FilenameParsingLayerUpdateController(getName(), parameters(), mCommunicator);
}

ActivityComponent *FilenameParsingGroundTruthLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<FilenameParsingActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

InputLayerNameParam *FilenameParsingGroundTruthLayer::createInputLayerNameParam() {
   return new InputLayerNameParam(getName(), parameters(), mCommunicator);
}

} // end namespace PV
