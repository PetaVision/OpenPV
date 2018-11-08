/*
 * VaryingHyPerConn.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#include "VaryingHyPerConn.hpp"
#include "IncrementingWeightUpdater.hpp"
#include <columns/HyPerCol.hpp>

namespace PV {

VaryingHyPerConn::VaryingHyPerConn(const char *name, PVParams *params, Communicator *comm)
      : HyPerConn() {
   initialize(name, params, comm);
}

VaryingHyPerConn::~VaryingHyPerConn() {}

void VaryingHyPerConn::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerConn::initialize(name, params, comm);
}

BaseWeightUpdater *VaryingHyPerConn::createWeightUpdater() {
   return new IncrementingWeightUpdater(name, parameters(), mCommunicator);
}

} // end of namespace PV block
