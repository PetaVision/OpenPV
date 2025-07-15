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
FilenameParsingLayer::FilenameParsingLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

FilenameParsingLayer::~FilenameParsingLayer() {}

void FilenameParsingLayer::fillComponentTable() {
   HyPerLayer::fillComponentTable();
   mInputLayerNameParam = createInputLayerNameParam();
   if (mInputLayerNameParam) {
      addUniqueComponent(mInputLayerNameParam);
   }
}

LayerInputBuffer *FilenameParsingLayer::createLayerInput() { return nullptr; }

LayerUpdateController *FilenameParsingLayer::createLayerUpdateController() {
   return new FilenameParsingLayerUpdateController(mParamsIO, mCommunicator);
}

ActivityComponent *FilenameParsingLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<FilenameParsingActivityBuffer>(
         mParamsIO, mCommunicator);
}

InputLayerNameParam *FilenameParsingLayer::createInputLayerNameParam() {
   return new InputLayerNameParam(mParamsIO, mCommunicator);
}

} // end namespace PV
