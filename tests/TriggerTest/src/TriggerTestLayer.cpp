/*
 * TriggerTestLayer.cpp
 * Author: slundquist
 */

#include "TriggerTestLayer.hpp"
#include <utils/PVLog.hpp>

namespace PV {
TriggerTestLayer::TriggerTestLayer(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

Response::Status TriggerTestLayer::checkUpdateState(double simTime, double deltaTime) {
   // 4 different layers
   // No trigger, always update
   InfoLog().printf(
         "%s: time=%f, dt=%f, needUpdate=%d, triggerOffset=%f\n",
         name,
         simTime,
         deltaTime,
         mLayerUpdateController->needUpdate(simTime, deltaTime),
         mLayerUpdateController->getTriggerOffset());
   if (strcmp(name, "notrigger") == 0) {
      FatalIf(
            mLayerUpdateController->needUpdate(simTime, deltaTime) == false,
            "Test failed at %s. Expected true, found false.\n",
            getName());
   }
   // Trigger with offset of 0, assuming display period is 5
   else if (strcmp(name, "trigger0") == 0) {
      if (((int)simTime - 1) % 5 == 0) {
         FatalIf(
               mLayerUpdateController->needUpdate(simTime, deltaTime) == false,
               "Test failed at %s. Expected true, found false.\n",
               getName());
      }
      else {
         FatalIf(
               mLayerUpdateController->needUpdate(simTime, deltaTime) == true,
               "Test failed at %s. Expected false, found true.\n",
               getName());
      }
   }
   // Trigger with offset of 1, assuming display period is 5
   else if (strcmp(name, "trigger1") == 0) {
      if (((int)simTime) % 5 == 0) {
         FatalIf(
               mLayerUpdateController->needUpdate(simTime, deltaTime) == false,
               "Test failed at %s. Expected true, found false.\n",
               getName());
      }
      else {
         FatalIf(
               mLayerUpdateController->needUpdate(simTime, deltaTime) == true,
               "Test failed at %s. Expected false, found true.\n",
               getName());
      }
   }
   // Trigger with offset of 1, assuming display period is 5
   else if (strcmp(name, "trigger2") == 0) {
      if (((int)simTime + 1) % 5 == 0) {
         FatalIf(
               mLayerUpdateController->needUpdate(simTime, deltaTime) == false,
               "Test failed at %s. Expected true, found false.\n",
               getName());
      }
      else {
         FatalIf(
               mLayerUpdateController->needUpdate(simTime, deltaTime) == true,
               "Test failed at %s. Expected false, found true.\n",
               getName());
      }
   }
   return HyPerLayer::checkUpdateState(simTime, deltaTime);
}
}
