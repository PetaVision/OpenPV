/*
 * PlasticTestUpdater.cpp
 *
 *  Created on: Oct 19, 2011
 *      Author: pschultz
 */

#include "PlasticTestUpdater.hpp"
#include <columns/HyPerCol.hpp>

namespace PV {

PlasticTestUpdater::PlasticTestUpdater(const char *name, HyPerCol *hc) : HebbianUpdater() {
   HebbianUpdater::initialize(name, hc);
}

float PlasticTestUpdater::updateRule_dW(float pre, float post) { return pre - post; }

PlasticTestUpdater::~PlasticTestUpdater() {}

} /* namespace PV */
