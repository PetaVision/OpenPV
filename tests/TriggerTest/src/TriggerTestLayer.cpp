/*
 * TriggerTestLayer.cpp
 * Author: slundquist
 */

#include "TriggerTestLayer.hpp"
#include <utils/PVLog.hpp>

namespace PV {
TriggerTestLayer::TriggerTestLayer(const char * name, HyPerCol * hc)
{
   HyPerLayer::initialize(name, hc);
}

int TriggerTestLayer::updateState(double time, double dt){
   //4 different layers
   //No trigger, always update
   pvInfo().printf("%s: time=%f, dt=%f, needUpdate=%d\n", name, time, dt, needUpdate(time, dt));
   if(strcmp(name, "notrigger") == 0){
      pvErrorIf(!(needUpdate(time, dt) == true), "Test failed.\n");
   }
   //Trigger with offset of 0, assuming display period is 5
   else if(strcmp(name, "trigger0") == 0){
      if(((int)time-1) % 5 == 0){
         pvErrorIf(!(needUpdate(time, dt) == true), "Test failed.\n");
      }
      else{
         pvErrorIf(!(needUpdate(time, dt) == false), "Test failed.\n");
      }
   }
   //Trigger with offset of 1, assuming display period is 5
   else if(strcmp(name, "trigger1") == 0){
      if(((int)time) % 5 == 0){
         pvErrorIf(!(needUpdate(time, dt) == true), "Test failed.\n");
      }
      else{
         pvErrorIf(!(needUpdate(time, dt) == false), "Test failed.\n");
      }
   }
   //Trigger with offset of 1, assuming display period is 5
   else if(strcmp(name, "trigger2") == 0){
      if(((int)time+1) % 5 == 0){
         pvErrorIf(!(needUpdate(time, dt) == true), "Test failed.\n");
      }
      else{
         pvErrorIf(!(needUpdate(time, dt) == false), "Test failed.\n");
      }
   }
   return HyPerLayer::updateState(time, dt);
}

}
