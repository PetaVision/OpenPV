/*
 * TriggerTestConn.cpp
 * Author: slundquist
 */

#include "TriggerTestConn.hpp"
#include <utils/PVLog.hpp>

namespace PV {
TriggerTestConn::TriggerTestConn(const char *name, HyPerCol *hc) {
   HyPerConn::initialize(name, hc);
}

int TriggerTestConn::updateState(double time, double dt) {
   // 4 different layers
   // No trigger, always update
   InfoLog().printf("TriggerTestConn %s updating at time %f\n", getName(), time);
   if (strcmp(name, "inputToNoTrigger") == 0 || strcmp(name, "inputToNoPeriod") == 0) {
      FatalIf(
            !(needUpdate(time, dt) == true),
            "Test failed at %s. Expected true, found false.\n",
            getName());
   }
   // Trigger with offset of 0, assuming display period is 5
   else if (strcmp(name, "inputToTrigger0") == 0 || strcmp(name, "inputToPeriod0") == 0) {
      if (((int)time - 1) % 5 == 0) {
         FatalIf(
               !(needUpdate(time, dt) == true),
               "Test failed at %s. Expected true, found false.\n",
               getName());
      }
      else {
         FatalIf(
               !(needUpdate(time, dt) == false),
               "Test failed at %s. Expected false, found true.\n",
               getName());
      }
   }
   // Trigger with offset of 1, assuming display period is 5
   else if (strcmp(name, "inputToTrigger1") == 0 || strcmp(name, "inputToPeriod1") == 0) {
      if (((int)time) % 5 == 0) {
         FatalIf(
               !(needUpdate(time, dt) == true),
               "Test failed at %s. Expected true, found false.\n",
               getName());
      }
      else {
         FatalIf(
               !(needUpdate(time, dt) == false),
               "Test failed at %s. Expected false, found true.\n",
               getName());
      }
   }
   // Trigger with offset of 2, assuming display period is 5
   else if (strcmp(name, "inputToTrigger2") == 0 || strcmp(name, "inputToPeriod2") == 0) {
      if (((int)time + 1) % 5 == 0) {
         FatalIf(
               !(needUpdate(time, dt) == true),
               "Test failed at %s. Expected true, found false.\n",
               getName());
      }
      else {
         FatalIf(
               !(needUpdate(time, dt) == false),
               "Test failed at %s. Expected false, found true.\n",
               getName());
      }
   }
   return HyPerConn::updateState(time, dt);
}
}
