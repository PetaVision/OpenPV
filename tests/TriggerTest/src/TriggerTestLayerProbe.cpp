/*
 * TriggerTestLayer.cpp
 * Author: slundquist
 */

#include "TriggerTestLayerProbe.hpp"
#include <columns/HyPerCol.hpp>
#include <utils/PVLog.hpp>

namespace PV {
TriggerTestLayerProbe::TriggerTestLayerProbe(const char *name, HyPerCol *hc) {
   LayerProbe::initialize(name, hc);
}

void TriggerTestLayerProbe::calcValues(double timevalue) {
   double v             = needUpdate(timevalue, parent->getDeltaTime()) ? 1.0 : 0.0;
   double *valuesBuffer = this->getValuesBuffer();
   for (int n = 0; n < this->getNumValues(); n++) {
      valuesBuffer[n] = v;
   }
}

Response::Status TriggerTestLayerProbe::outputStateWrapper(double time, double dt) {
   // Time 0 is initialization, doesn't matter if it updates or not
   if (time < dt / 2) {
      return LayerProbe::outputStateWrapper(time, dt);
   }

   // 4 different layers
   // No trigger, always update
   const char *name = getName();
   getValues(time);
   FatalIf(!(this->getNumValues() > 0), "Test failed.\n");
   int updateNeeded = (int)getValuesBuffer()[0];
   InfoLog().printf(
         "%s: time=%f, dt=%f, needUpdate=%d, triggerOffset=%f\n",
         name,
         time,
         dt,
         updateNeeded,
         triggerOffset);
   if (strcmp(name, "notriggerlayerprobe") == 0) {
      FatalIf(!(updateNeeded == 1), "Test failed at %s. Expected true, found false.\n", getName());
   }
   // Trigger with offset of 0, assuming display period is 5
   else if (strcmp(name, "trigger0layerprobe") == 0) {
      if (((int)time - 1) % 5 == 0) {
         FatalIf(
               !(updateNeeded == 1), "Test failed at %s. Expected true, found false.\n", getName());
      }
      else {
         FatalIf(
               !(updateNeeded == 0), "Test failed at %s. Expected false, found true.\n", getName());
      }
   }
   // Trigger with offset of 1, assuming display period is 5
   else if (strcmp(name, "trigger1layerprobe") == 0) {
      if (((int)time) % 5 == 0) {
         FatalIf(
               !(updateNeeded == 1), "Test failed at %s. Expected true, found false.\n", getName());
      }
      else {
         FatalIf(
               !(updateNeeded == 0), "Test failed at %s. Expected false, found true.\n", getName());
      }
   }
   // Trigger with offset of 1, assuming display period is 5
   else if (strcmp(name, "trigger2layerprobe") == 0) {
      if (((int)time + 1) % 5 == 0) {
         FatalIf(
               !(updateNeeded == 1), "Test failed at %s. Expected true, found false.\n", getName());
      }
      else {
         FatalIf(
               !(updateNeeded == 0), "Test failed at %s. Expected false, found true.\n", getName());
      }
   }
   return LayerProbe::outputStateWrapper(time, dt);
}

Response::Status TriggerTestLayerProbe::outputState(double timef) { return Response::SUCCESS; }

} // namespace PV
