/*
 * MarginWidthTest.cpp
 *
 *  Created on: Jul 24, 2013
 *      Author: Pete Schultz
 *
 *
 */

#include <columns/buildandrun.hpp>

int custominit(HyPerCol * hc, int argc, char **argv);
// custominit is for doing things after the HyPerCol has been built but before the run method is called.

int customexit(HyPerCol * hc, int argc, char **argv);
// customexit is for doing things after the run completes but before the HyPerCol is deleted.

void * customgroup(const char * name, const char * groupname, HyPerCol * hc);
// customgroups is for adding objects not supported by build().

bool checkHalo(PVHalo const * halo, int lt, int rt, int dn, int up);

int main(int argc, char * argv[]) {

   int status;
   status = buildandrun(argc, argv, NULL, &customexit, &customgroup);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int custominit(HyPerCol * hc, int argc, char ** argv) {
   PVHalo check;
   assert(checkHalo(&hc->getLayerFromName("MarginsEqualImage")->getLayerLoc()->halo, 0, 0, 0, 0));
   assert(checkHalo(&hc->getLayerFromName("XMarginLargerImage")->getLayerLoc()->halo, 0, 0, 0, 0));
   assert(checkHalo(&hc->getLayerFromName("YMarginLargerImage")->getLayerLoc()->halo, 0, 0, 0, 0));
   assert(checkHalo(&hc->getLayerFromName("MultipleConnImage")->getLayerLoc()->halo, 0, 0, 0, 0));
   return PV_SUCCESS;
}

int customexit(HyPerCol * hc, int argc, char ** argv) {
   assert(checkHalo(&hc->getLayerFromName("MarginsEqualImage")->getLayerLoc()->halo, 2, 2, 2, 2));
   assert(checkHalo(&hc->getLayerFromName("XMarginLargerImage")->getLayerLoc()->halo,3, 3, 1, 1));
   assert(checkHalo(&hc->getLayerFromName("YMarginLargerImage")->getLayerLoc()->halo,1, 1, 3, 3));
   assert(checkHalo(&hc->getLayerFromName("MultipleConnImage")->getLayerLoc()->halo,3, 3, 3, 3));
   return PV_SUCCESS;
}

void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   void * addedGroup = NULL;
   return addedGroup;
}

bool checkHalo(PVHalo const * halo, int lt, int rt, int dn, int up) {
   return
         halo->lt==lt &&
         halo->rt==rt &&
         halo->dn==dn &&
         halo->up==up;
}


