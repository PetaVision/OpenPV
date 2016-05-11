/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>
#include "ResetStateOnTriggerTestProbe.hpp"

int customexit(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("ResetStateOnTriggerTestProbe", createResetStateOnTriggerTestProbe);
   int status = buildandrun(&pv_initObj);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   HyPerLayer * l = hc->getLayerFromName("TestLayer");
   assert(l);
   assert(l->getNumProbes()==1);
   LayerProbe * p = l->getProbe(0);
   assert(!strcmp(p->getName(), "TestProbe"));
   ResetStateOnTriggerTestProbe * rsProbe = dynamic_cast<ResetStateOnTriggerTestProbe *>(p);
   assert(rsProbe);
   int status = PV_SUCCESS;
   if (rsProbe->getProbeStatus()) {
      if (hc->columnId()==0) {
         fprintf(stderr, "%s failed at time %f\n",
               argv[0], rsProbe->getFirstFailureTime());
      }
      status = PV_FAILURE;
   }
   return status;
}
