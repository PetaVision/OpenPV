/*
 * ShrunkenPatchTestLayer.cpp
 *
 *  Created on: Sep 27, 2011
 *      Author: gkenyon
 */

#include "ShrunkenPatchTestLayer.hpp"

#include "ShrunkenPatchTestActivityBuffer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"

namespace PV {

ShrunkenPatchTestLayer::ShrunkenPatchTestLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm)
      : HyPerLayer() {
   // ShrunkenPatchTestLayer has no member variables to initialize in initialize_base()
   initialize(paramsIO, comm);
}

void ShrunkenPatchTestLayer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *ShrunkenPatchTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ShrunkenPatchTestActivityBuffer>(
         mParamsIO, mCommunicator);
}

} /* namespace PV */
