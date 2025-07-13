/* MomentumConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "MomentumConn.hpp"
#include "weightupdaters/MomentumUpdater.hpp"

namespace PV {

MomentumConn::MomentumConn(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

MomentumConn::MomentumConn() {}

MomentumConn::~MomentumConn() {}

void MomentumConn::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerConn::initialize(params, defaults, comm);
}

BaseWeightUpdater *MomentumConn::createWeightUpdater() {
   return new MomentumUpdater(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // namespace PV
