/*
 * PlasticTestUpdater.cpp
 *
 *  Created on: Oct 19, 2011
 *      Author: pschultz
 */

#include "PlasticTestUpdater.hpp"
#include <columns/HyPerCol.hpp>

namespace PV {

PlasticTestUpdater::PlasticTestUpdater(const char *name, PVParams *params, Communicator *comm)
      : HebbianUpdater() {
   HebbianUpdater::initialize(name, params, comm);
}

float PlasticTestUpdater::updateRule_dW(float pre, float post) { return pre - post; }

PlasticTestUpdater::~PlasticTestUpdater() {}

} /* namespace PV */
