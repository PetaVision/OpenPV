/*
 * TriggerTestConn.cpp
 * Author: slundquist
 */

#include "TriggerTestConn.hpp"
#include "TriggerTestUpdater.hpp"

namespace PV {
TriggerTestConn::TriggerTestConn(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerConn::initialize(paramsIO, comm);
}

BaseWeightUpdater *TriggerTestConn::createWeightUpdater() {
   return new TriggerTestUpdater(mParamsIO, mCommunicator);
}

} // namespace PV
