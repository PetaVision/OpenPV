/*
 * MarginWidthTest.cpp
 *
 *  Created on: Jul 24, 2013
 *      Author: Pete Schultz
 *
 *
 */

#include <columns/buildandrun.hpp>
#include <layers/HyPerLayer.hpp>

int customexit(HyPerCol *hc, int argc, char **argv);
// customexit is for doing things after the run completes but before the HyPerCol is deleted.
// For this test, we check whether the layer margins have the expected values.

bool checkHalo(HyPerCol *hc, std::string const &layerName, int lt, int rt, int dn, int up);

int main(int argc, char *argv[]) {

   int status;
   status = buildandrun(argc, argv, nullptr, &customexit);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol *hc, int argc, char **argv) {
   FatalIf(!checkHalo(hc, std::string("MarginsEqualImage"), 2, 2, 2, 2), "Test failed.\n");
   FatalIf(!checkHalo(hc, std::string("XMarginLargerImage"), 3, 3, 1, 1), "Test failed.\n");
   FatalIf(!checkHalo(hc, std::string("YMarginLargerImage"), 1, 1, 3, 3), "Test failed.\n");
   FatalIf(!checkHalo(hc, std::string("MultipleConnImage"), 3, 3, 3, 3), "Test failed.\n");
   return PV_SUCCESS;
}

bool checkHalo(HyPerCol *hc, std::string const &layerName, int lt, int rt, int dn, int up) {
   HyPerLayer *layer  = dynamic_cast<HyPerLayer *>(hc->getObjectFromName(layerName));
   PVHalo const &halo = layer->getLayerLoc()->halo;
   bool passed        = true;
   if (halo.lt != lt) {
      ErrorLog().printf(
            "%s has left margin of %d, should be %d.\n", layerName.c_str(), halo.lt, lt);
      passed = false;
   }
   if (halo.rt != rt) {
      ErrorLog().printf(
            "%s has left margin of %d, should be %d.\n", layerName.c_str(), halo.rt, rt);
      passed = false;
   }
   if (halo.dn != dn) {
      ErrorLog().printf(
            "%s has left margin of %d, should be %d.\n", layerName.c_str(), halo.dn, dn);
      passed = false;
   }
   if (halo.up != up) {
      ErrorLog().printf(
            "%s has left margin of %d, should be %d.\n", layerName.c_str(), halo.up, up);
      passed = false;
   }
   return passed;
}
