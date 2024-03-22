/*
 * TriggerTestUpdater.cpp
 *
 *  Created on: Dec 8, 2017
 *      Author: peteschultz
 */

#include "TriggerTestUpdater.hpp"

namespace PV {

TriggerTestUpdater::TriggerTestUpdater(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   HebbianUpdater::initialize(name, params, comm);
}

void TriggerTestUpdater::updateState(double time, double dt) {
   // 4 different layers
   // No trigger, always update
   InfoLog().printf("TriggerTestUpdater %s updating at time %f\n", getName(), time);
   if (strcmp(getName(), "inputToNoTrigger") == 0 || strcmp(getName(), "inputToNoPeriod") == 0) {
      FatalIf(
            !(needUpdate(time, dt) == true),
            "Test failed at %s. Expected true, found false.\n",
            getName());
   }
   // Trigger with offset of 0, assuming display period is 5
   else if (strcmp(getName(), "inputToTrigger0") == 0 || strcmp(getName(), "inputToPeriod0") == 0) {
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
   else if (strcmp(getName(), "inputToTrigger1") == 0 || strcmp(getName(), "inputToPeriod1") == 0) {
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
   else if (strcmp(getName(), "inputToTrigger2") == 0 || strcmp(getName(), "inputToPeriod2") == 0) {
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
   HebbianUpdater::updateState(time, dt);
}

} // namespace PV
