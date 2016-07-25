/*
 * TriggerTestLayer.cpp
 * Author: slundquist
 */

#include "TriggerTestLayer.hpp"
#include <assert.h>

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
      assert(needUpdate(time, dt) == true);
   }
   //Trigger with offset of 0, assuming display period is 5
   else if(strcmp(name, "trigger0") == 0){
      if(((int)time-1) % 5 == 0){
         assert(needUpdate(time, dt) == true);
      }
      else{
         assert(needUpdate(time, dt) == false);
      }
   }
   //Trigger with offset of 1, assuming display period is 5
   else if(strcmp(name, "trigger1") == 0){
      if(((int)time) % 5 == 0){
         assert(needUpdate(time, dt) == true);
      }
      else{
         assert(needUpdate(time, dt) == false);
      }
   }
   //Trigger with offset of 1, assuming display period is 5
   else if(strcmp(name, "trigger2") == 0){
      if(((int)time+1) % 5 == 0){
         assert(needUpdate(time, dt) == true);
      }
      else{
         assert(needUpdate(time, dt) == false);
      }
   }
   return HyPerLayer::updateState(time, dt);
}

BaseObject * createTriggerTestLayer(char const * name, HyPerCol * hc) {
   return hc ? new TriggerTestLayer(name, hc) : NULL;
}

}
