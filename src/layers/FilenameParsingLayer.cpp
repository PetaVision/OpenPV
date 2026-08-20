/*
 * FilenameParsingLayer.cpp
 *
 *  Created on: Nov 10, 2014
 *      Author: wchavez
 */

#include "FilenameParsingLayer.hpp"

#include "components/ActivityComponentActivityOnly.hpp"
#include "components/FilenameParsingActivityBuffer.hpp"
#include "components/FilenameParsingLayerUpdateController.hpp"

namespace PV {
FilenameParsingLayer::FilenameParsingLayer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

FilenameParsingLayer::~FilenameParsingLayer() {}

void FilenameParsingLayer::fillComponentTable() {
   BaseLayer::fillComponentTable();
   mInputLayerNameParam = createInputLayerNameParam();
   if (mInputLayerNameParam) {
      addUniqueComponent(mInputLayerNameParam);
   }
}

LayerUpdateController *FilenameParsingLayer::createLayerUpdateController() {
   return new FilenameParsingLayerUpdateController(getName(), parameters(), mCommunicator);
}

ActivityComponent *FilenameParsingLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<FilenameParsingActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

InputLayerNameParam *FilenameParsingLayer::createInputLayerNameParam() {
   return new InputLayerNameParam(getName(), parameters(), mCommunicator);
}

} // end namespace PV
