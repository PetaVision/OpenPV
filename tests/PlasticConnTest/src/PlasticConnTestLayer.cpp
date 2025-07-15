/*
 * PlasticConnTestLayer.cpp
 *
 *  Created on: Oct 24, 2011
 *      Author: pschultz
 */

#include "PlasticConnTestLayer.hpp"

#include "PlasticConnTestActivityBuffer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"

namespace PV {

PlasticConnTestLayer::PlasticConnTestLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

void PlasticConnTestLayer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *PlasticConnTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<PlasticConnTestActivityBuffer>(
         mParamsIO, mCommunicator);
}

} /* namespace PV */
