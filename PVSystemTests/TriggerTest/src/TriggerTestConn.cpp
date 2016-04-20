/*
 * TriggerTestConn.cpp
 * Author: slundquist
 */

#include "TriggerTestConn.hpp"
#include <assert.h>

namespace PV {
TriggerTestConn::TriggerTestConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer)
{
   HyPerConn::initialize(name, hc, weightInitializer, weightNormalizer);
}

int TriggerTestConn::updateState(double time, double dt){
   //4 different layers
   //No trigger, always update
   if(strcmp(name, "inputToNoTrigger") == 0 || strcmp(name, "inputToNoPeriod") == 0){
      assert(needUpdate(time, dt) == true);
   }
   //Trigger with offset of 0, assuming display period is 5
   else if(strcmp(name, "inputToTrigger0") == 0 || strcmp(name, "inputToPeriod0") == 0){
      if(((int)time-1) % 5 == 0){
         assert(needUpdate(time, dt) == true);
      }
      else{
         assert(needUpdate(time, dt) == false);
      }
   }
   //Trigger with offset of 1, assuming display period is 5
   else if(strcmp(name, "inputToTrigger1") == 0 || strcmp(name, "inputToPeriod1") == 0){
      if(((int)time) % 5 == 0){
         assert(needUpdate(time, dt) == true);
      }
      else{
         assert(needUpdate(time, dt) == false);
      }
   }
   //Trigger with offset of 2, assuming display period is 5
   else if(strcmp(name, "inputToTrigger2") == 0 || strcmp(name, "inputToPeriod2") == 0){
      if(((int)time+1) % 5 == 0){
         assert(needUpdate(time, dt) == true);
      }
      else{
         assert(needUpdate(time, dt) == false);
      }
   }
   return HyPerConn::updateState(time, dt);
}

BaseObject * createTriggerTestConn(char const * name, HyPerCol * hc) {
   return hc ? new TriggerTestConn(name, hc) : NULL;
}

}
