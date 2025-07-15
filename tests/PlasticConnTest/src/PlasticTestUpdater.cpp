/*
 * PlasticTestUpdater.cpp
 *
 *  Created on: Oct 19, 2011
 *      Author: pschultz
 */

#include "PlasticTestUpdater.hpp"

namespace PV {

PlasticTestUpdater::PlasticTestUpdater(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm)
      : HebbianUpdater() {
   HebbianUpdater::initialize(paramsIO, comm);
}

float PlasticTestUpdater::updateRule_dW(float pre, float post) { return pre - post; }

PlasticTestUpdater::~PlasticTestUpdater() {}

} /* namespace PV */
