/*
 * Retina.cpp
 *
 *  Created on: Jul 29, 2008
 *
 */

#include "Retina.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/RetinaActivityBuffer.hpp"

namespace PV {

Retina::Retina(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

Retina::Retina() {}

Retina::~Retina() {}

void Retina::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *Retina::createActivityComponent() {
   return new ActivityComponentActivityOnly<RetinaActivityBuffer>(
         mParamsIO, mCommunicator);
}

} // namespace PV
