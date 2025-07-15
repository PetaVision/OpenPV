/*
 * MPITestLayer.cpp
 *
 *  Created on: Sep 27, 2011
 *      Author: gkenyon
 */

#include "MPITestLayer.hpp"

#include "MPITestActivityBuffer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"

namespace PV {

MPITestLayer::MPITestLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm)
      : HyPerLayer() {
   // MPITestLayer has no member variables to initialize in initialize_base()
   initialize(paramsIO, comm);
}

void MPITestLayer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *MPITestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<MPITestActivityBuffer>(
         mParamsIO, mCommunicator);
}

} /* namespace PV */
