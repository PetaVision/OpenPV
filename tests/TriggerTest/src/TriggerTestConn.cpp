/*
 * TriggerTestConn.cpp
 * Author: slundquist
 */

#include "TriggerTestConn.hpp"
#include "TriggerTestUpdater.hpp"

namespace PV {
TriggerTestConn::TriggerTestConn(const char *name, PVParams *params, Communicator const *comm) {
   HyPerConn::initialize(name, params, comm);
}

BaseWeightUpdater *TriggerTestConn::createWeightUpdater() {
   return new TriggerTestUpdater(getName(), parameters(), mCommunicator);
}

} // namespace PV
