/*
 * TriggerTestConn.cpp
 * Author: slundquist
 */

#include "TriggerTestConn.hpp"
#include "TriggerTestUpdater.hpp"

namespace PV {
TriggerTestConn::TriggerTestConn(const char *name, HyPerCol *hc) {
   HyPerConn::initialize(name, hc);
}

BaseWeightUpdater *TriggerTestConn::createWeightUpdater() {
   return new TriggerTestUpdater(name, parent);
}

} // namespace PV
