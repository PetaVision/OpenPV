/*
 * SparsityProbeTest.cpp
 * Author: slundquist
 */

#include "SparsityProbeTest.hpp"
#include <layers/HyPerLayer.hpp>
#include <assert.h>

namespace PV {

SparsityProbeTest::SparsityProbeTest(const char * name, HyPerCol * hc)
{
   SparsityLayerProbe::initLayerProbe(name , hc);
}

int SparsityProbeTest::outputState(double time){
   //Update probe
   int status = SparsityLayerProbe::outputState(time);
   double deltaTime = parentCol->getDeltaTime();
   //Skip on ts 0, initialization step
   if(fabs(time - 0) <= (deltaTime/2)){
      return status;
   }
   float actualVal;
   if(strcmp(probeName, "nnxProbe") == 0){
      actualVal = time / 64;
   }
   else{
      actualVal = (time * .5)/64;
   }

   //Sparsity based on time
   //.01 roundoff error
   assert(getSparsity() - actualVal < .01);
   return status;
}

}
