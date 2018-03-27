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

int custominit(HyPerCol *hc, int argc, char **argv);
// custominit is for doing things after the HyPerCol has been built but before the run method is
// called.

int customexit(HyPerCol *hc, int argc, char **argv);
// customexit is for doing things after the run completes but before the HyPerCol is deleted.

bool checkHalo(HyPerCol *hc, std::string const &layerName, int lt, int rt, int dn, int up);

int main(int argc, char *argv[]) {

   int status;
   status = buildandrun(argc, argv, &custominit, &customexit);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int custominit(HyPerCol *hc, int argc, char **argv) {
   PVHalo check;
   FatalIf(!checkHalo(hc, std::string("MarginsEqualImage"), 0, 0, 0, 0), "Test failed.\n");
   FatalIf(!checkHalo(hc, std::string("XMarginLargerImage"), 0, 0, 0, 0), "Test failed.\n");
   FatalIf(!checkHalo(hc, std::string("YMarginLargerImage"), 0, 0, 0, 0), "Test failed.\n");
   FatalIf(!checkHalo(hc, std::string("MultipleConnImage"), 0, 0, 0, 0), "Test failed.\n");
   return PV_SUCCESS;
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
   return halo.lt == lt && halo.rt == rt && halo.dn == dn && halo.up == up;
}
