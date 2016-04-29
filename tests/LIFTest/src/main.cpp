/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "AverageRateConn.hpp"
#include "LIFTestProbe.hpp"

int customexit(HyPerCol * hc, int argc, char * argv[]);
// customexit is called after each entry in the parameter sweep (or once at the end if there are no parameter sweeps) and before the HyPerCol is deleted.


int main(int argc, char * argv[]) {

   int status;
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("AverageRateConn", createAverageRateConn);
   pv_initObj.registerKeyword("LIFTestProbe", createLIFTestProbe);
   status = buildandrun(&pv_initObj, NULL/*custominit*/, &customexit);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   HyPerLayer * spikecount = hc->getLayerFromName("LIFGapTestSpikeCounter");
   int status = spikecount != NULL ? PV_SUCCESS : PV_FAILURE;
   if (status != PV_SUCCESS) {
      if (hc->icCommunicator()->commRank()==0) {
         fprintf(stderr, "Error:  No layer named \"LIFGapTestSpikeCounter\"");
      }
      status = PV_FAILURE;
   }
   return status;
}
