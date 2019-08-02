/*
 * FirmThresholdCostTest.cpp
 *
 */

#include <columns/buildandrun.hpp>

int main(int argc, char *argv[]) {
   auto *pv_initObj = new PV::PV_Init(&argc, &argv, false /*do not allow unrecognized arguments*/);
   auto *hc = new PV::HyPerCol(pv_initObj);

   hc->run();

   delete hc;
   delete pv_initObj;
}
