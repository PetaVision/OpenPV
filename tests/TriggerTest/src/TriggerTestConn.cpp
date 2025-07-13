/*
 * TriggerTestConn.cpp
 * Author: slundquist
 */

#include "TriggerTestConn.hpp"
#include "TriggerTestUpdater.hpp"

namespace PV {
TriggerTestConn::TriggerTestConn(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerConn::initialize(params, defaults, comm);
}

BaseWeightUpdater *TriggerTestConn::createWeightUpdater() {
   return new TriggerTestUpdater(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // namespace PV
