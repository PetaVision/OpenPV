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
      const char *name,
      PVParams *params,
      Communicator const *comm)
      : HyPerLayer() {
   // ShrunkenPatchTestLayer has no member variables to initialize in initialize_base()
   initialize(name, params, comm);
}

void ShrunkenPatchTestLayer::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *ShrunkenPatchTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ShrunkenPatchTestActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} /* namespace PV */
