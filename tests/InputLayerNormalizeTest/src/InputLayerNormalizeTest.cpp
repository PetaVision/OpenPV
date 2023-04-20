/*
 * pv.cpp
 *
 */

#include "CheckStatsProbe.hpp"
#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>
#include <columns/Factory.hpp>

int main(int argc, char *argv[]) {
   auto pv_initObj = new PV::PV_Init(&argc, &argv, false /*do not allow unrecognized arguments*/);
   pv_initObj->registerKeyword("CheckStatsProbe", PV::Factory::create<CheckStatsProbe>);
   int status = buildandrun(pv_initObj);
   delete pv_initObj;
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
