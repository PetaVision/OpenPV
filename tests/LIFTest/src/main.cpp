/*
 * pv.cpp
 *
 */

#include "AverageRateConn.hpp"
#include "LIFTestProbe.hpp"
#include <columns/buildandrun.hpp>

int customexit(HyPerCol *hc, int argc, char *argv[]);
// customexit is called after each entry in the parameter sweep (or once at the end if there are no
// parameter sweeps) and before the HyPerCol is deleted.

int main(int argc, char *argv[]) {

   int status;
   PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("AverageRateConn", Factory::create<AverageRateConn>);
   pv_initObj.registerKeyword("LIFTestProbe", Factory::create<LIFTestProbe>);
   status = buildandrun(&pv_initObj, NULL /*custominit*/, &customexit);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol *hc, int argc, char *argv[]) {
   HyPerLayer *spikecount =
         dynamic_cast<HyPerLayer *>(hc->getObjectFromName("LIFGapTestSpikeCounter"));
   int status = spikecount != NULL ? PV_SUCCESS : PV_FAILURE;
   if (status != PV_SUCCESS) {
      if (hc->getCommunicator()->commRank() == 0) {
         Fatal().printf("Error:  No layer named \"LIFGapTestSpikeCounter\"");
      }
   }
   return status;
}
