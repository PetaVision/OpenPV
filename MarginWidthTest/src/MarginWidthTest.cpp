/*
 * MarginWidthTest.cpp
 *
 *  Created on: Jul 24, 2013
 *      Author: Pete Schultz
 *
 *
 */

#include "../../PetaVision/src/columns/buildandrun.hpp"

int custominit(HyPerCol * hc, int argc, char **argv);
// custominit is for doing things after the HyPerCol has been built but before the run method is called.

int customexit(HyPerCol * hc, int argc, char **argv);
// customexit is for doing things after the run completes but before the HyPerCol is deleted.

void * customgroup(const char * name, const char * groupname, HyPerCol * hc);
// customgroups is for adding objects not supported by build().

int main(int argc, char * argv[]) {

   int status;
   status = buildandrun(argc, argv, NULL, &customexit, &customgroup);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int custominit(HyPerCol * hc, int argc, char ** argv) {
   assert(hc->getLayerFromName("OneToOneCenterFirstInput")->getLayerLoc()->nb==0);
   assert(hc->getLayerFromName("OneToOneSurroundFirstInput")->getLayerLoc()->nb==0);
   assert(hc->getLayerFromName("ManyToOneCenterFirstInput")->getLayerLoc()->nb==0);
   assert(hc->getLayerFromName("ManyToOneSurroundFirstInput")->getLayerLoc()->nb==0);
   assert(hc->getLayerFromName("OneToManyCenterFirstInput")->getLayerLoc()->nb==0);
   assert(hc->getLayerFromName("OneToManySurroundFirstInput")->getLayerLoc()->nb==0);
   return PV_SUCCESS;
}

int customexit(HyPerCol * hc, int argc, char ** argv) {
   assert(hc->getLayerFromName("OneToOneCenterFirstInput")->getLayerLoc()->nb==4);
   assert(hc->getLayerFromName("OneToOneSurroundFirstInput")->getLayerLoc()->nb==4);
   assert(hc->getLayerFromName("ManyToOneCenterFirstInput")->getLayerLoc()->nb==16);
   assert(hc->getLayerFromName("ManyToOneSurroundFirstInput")->getLayerLoc()->nb==16);
   assert(hc->getLayerFromName("OneToManyCenterFirstInput")->getLayerLoc()->nb==4);
   assert(hc->getLayerFromName("OneToManySurroundFirstInput")->getLayerLoc()->nb==4);
   return PV_SUCCESS;
}

void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   void * addedGroup = NULL;
   return addedGroup;
}




