/*
 * PlasticTestConn.cpp
 *
 *  Created on: Oct 19, 2011
 *      Author: pschultz
 */

#include "PlasticTestConn.hpp"
#include "PlasticTestUpdater.hpp"

namespace PV {

PlasticTestConn::PlasticTestConn(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm)
      : HyPerConn() {
   HyPerConn::initialize(params, defaults, comm);
}

BaseWeightUpdater *PlasticTestConn::createWeightUpdater() {
   return new PlasticTestUpdater(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

PlasticTestConn::~PlasticTestConn() {}

} /* namespace PV */
