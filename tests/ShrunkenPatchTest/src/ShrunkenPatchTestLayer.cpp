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

ShrunkenPatchTestLayer::ShrunkenPatchTestLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm)
      : HyPerLayer() {
   // ShrunkenPatchTestLayer has no member variables to initialize in initialize_base()
   initialize(params, defaults, comm);
}

void ShrunkenPatchTestLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

ActivityComponent *ShrunkenPatchTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ShrunkenPatchTestActivityBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} /* namespace PV */
