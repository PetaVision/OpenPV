/*
 * PlasticTestUpdater.cpp
 *
 *  Created on: Oct 19, 2011
 *      Author: pschultz
 */

#include "PlasticTestUpdater.hpp"

namespace PV {

PlasticTestUpdater::PlasticTestUpdater(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm)
      : HebbianUpdater() {
   HebbianUpdater::initialize(params, defaults, comm);
}

float PlasticTestUpdater::updateRule_dW(float pre, float post) { return pre - post; }

PlasticTestUpdater::~PlasticTestUpdater() {}

} /* namespace PV */
