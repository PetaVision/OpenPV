/*
 * RequireAllZeroActivityProbeTest.cpp
 */

#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>
#include <cstdlib>
#include <exception>
#include <include/pv_common.h>
#include <utils/PVLog.hpp>

using PV::PV_Init;

int run(PV_Init *pv_init);
void runParams(PV_Init *pv_init, char const *paramsPath, int expectedStatus);

int main(int argc, char *argv[]) {
   auto *pv_init = new PV_Init(&argc, &argv, false /* whether to allow unrecognized arguments */);
   FatalIf(
         pv_init->getParams() != nullptr,
         "This test should be run without a params file argument. "
         "The necessary params files are hard-coded.\n");
   int status = run(pv_init);
   delete pv_init;
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int run(PV_Init *pv_init) {
   runParams(pv_init, "input/allzeros.params", PV_SUCCESS);
   runParams(pv_init, "input/nonzeropresent.params", PV_FAILURE);

   // runParams() will call Fatal(), so if we get here, the test passed.
   return PV_SUCCESS;
}

void runParams(PV_Init *pv_init, char const *paramsPath, int expectedStatus) {
   pv_init->setParams(paramsPath);
   PV::HyPerCol hc(pv_init);
   int status = PV_SUCCESS;
   try {
      status = hc.run();
   } catch (std::exception const &e) {
      status = PV_FAILURE;
   }
   FatalIf(
         expectedStatus == PV_SUCCESS and status != PV_SUCCESS,
         "Running with \"%s\" failed when it should have succeeded.\n",
         paramsPath);
   FatalIf(
         expectedStatus != PV_SUCCESS and status == PV_SUCCESS,
         "Running with \"%s\" returned successfully when it should have failed.\n",
         paramsPath);
}
