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
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
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
   return new FilenameParsingLayerUpdateController(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

ActivityComponent *FilenameParsingLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<FilenameParsingActivityBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

InputLayerNameParam *FilenameParsingLayer::createInputLayerNameParam() {
   return new InputLayerNameParam(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
