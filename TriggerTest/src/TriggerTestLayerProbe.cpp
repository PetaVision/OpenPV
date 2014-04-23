/*
 * TriggerTestLayer.cpp
 * Author: slundquist
 */

#include "TriggerTestLayerProbe.hpp"
#include <assert.h>

namespace PV {
TriggerTestLayerProbe::TriggerTestLayerProbe(const char * name, HyPerCol * hc)
{
   LayerProbe::initLayerProbe(name , hc);
}

int TriggerTestLayerProbe::outputStateWrapper(double time, double dt){
   //Time 0 is initialization, doesn't matter if it updates or not
   if(time < dt/2){
      return LayerProbe::outputStateWrapper(time, dt);
   }

   //4 different layers
   //No trigger, always update
   const char * name = getProbeName();
   fprintf(stderr, "%s: time=%f, dt=%f, needUpdate=%d\n", name, time, dt, needUpdate(time, dt));
   if(strcmp(name, "notriggerlayerprobe") == 0){
      assert(needUpdate(time, dt) == true);
   }
   //Trigger with offset of 0, assuming display period is 5
   else if(strcmp(name, "trigger0layerprobe") == 0){
      if(((int)time-1) % 5 == 0){
         assert(needUpdate(time, dt) == true);
      }
      else{
         assert(needUpdate(time, dt) == false);
      }
   }
   //Trigger with offset of 1, assuming display period is 5
   else if(strcmp(name, "trigger1layerprobe") == 0){
      if(((int)time) % 5 == 0){
         assert(needUpdate(time, dt) == true);
      }
      else{
         assert(needUpdate(time, dt) == false);
      }
   }
   //Trigger with offset of 1, assuming display period is 5
   else if(strcmp(name, "trigger2layerprobe") == 0){
      if(((int)time+1) % 5 == 0){
         assert(needUpdate(time, dt) == true);
      }
      else{
         assert(needUpdate(time, dt) == false);
      }
   }
   return LayerProbe::outputStateWrapper(time, dt);
}
int TriggerTestLayerProbe::outputState(double timef)
{
   return 0;
}
}
