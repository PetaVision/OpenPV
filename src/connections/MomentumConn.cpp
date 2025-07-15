/* MomentumConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "MomentumConn.hpp"
#include "weightupdaters/MomentumUpdater.hpp"

namespace PV {

MomentumConn::MomentumConn(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

MomentumConn::MomentumConn() {}

MomentumConn::~MomentumConn() {}

void MomentumConn::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerConn::initialize(paramsIO, comm);
}

BaseWeightUpdater *MomentumConn::createWeightUpdater() {
   return new MomentumUpdater(mParamsIO, mCommunicator);
}

} // namespace PV
