/* MomentumConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "MomentumConn.hpp"
#include "weightupdaters/MomentumUpdater.hpp"

namespace PV {

MomentumConn::MomentumConn(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

MomentumConn::MomentumConn() {}

MomentumConn::~MomentumConn() {}

void MomentumConn::initialize(char const *name, PVParams *params, Communicator const *comm) {
   HyPerConn::initialize(name, params, comm);
}

BaseWeightUpdater *MomentumConn::createWeightUpdater() {
   return new MomentumUpdater(name, parameters(), mCommunicator);
}

} // namespace PV
