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

Retina::Retina(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

Retina::Retina() {}

Retina::~Retina() {}

void Retina::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

ActivityComponent *Retina::createActivityComponent() {
   return new ActivityComponentActivityOnly<RetinaActivityBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // namespace PV
