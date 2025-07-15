/*
 * PlasticTestConn.cpp
 *
 *  Created on: Oct 19, 2011
 *      Author: pschultz
 */

#include "PlasticTestConn.hpp"
#include "PlasticTestUpdater.hpp"

namespace PV {

PlasticTestConn::PlasticTestConn(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm)
      : HyPerConn() {
   HyPerConn::initialize(paramsIO, comm);
}

BaseWeightUpdater *PlasticTestConn::createWeightUpdater() {
   return new PlasticTestUpdater(mParamsIO, mCommunicator);
}

PlasticTestConn::~PlasticTestConn() {}

} /* namespace PV */
