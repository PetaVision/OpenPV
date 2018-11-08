/*
 * PlasticTestConn.cpp
 *
 *  Created on: Oct 19, 2011
 *      Author: pschultz
 */

#include "PlasticTestConn.hpp"
#include "PlasticTestUpdater.hpp"

namespace PV {

PlasticTestConn::PlasticTestConn(const char *name, PVParams *params, Communicator *comm)
      : HyPerConn() {
   HyPerConn::initialize(name, params, comm);
}

BaseWeightUpdater *PlasticTestConn::createWeightUpdater() {
   return new PlasticTestUpdater(name, parameters(), mCommunicator);
}

PlasticTestConn::~PlasticTestConn() {}

} /* namespace PV */
