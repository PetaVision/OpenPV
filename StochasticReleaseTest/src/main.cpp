/*
 * main.cpp
 *
 * Minimal interface to PetaVision
 */


#include <columns/buildandrun.hpp>
#include "StochasticReleaseTestProbe.hpp"

void * customgroup(const char * name, const char * groupname, HyPerCol * hc);
// customgroups is for adding objects not supported by build().

int main(int argc, char * argv[]) {

   int status;
   status = buildandrun(argc, argv, NULL, NULL, &customgroup);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   void * addedGroup = NULL;
   if (!strcmp(keyword, "StochasticReleaseTestProbe")) {
      addedGroup = new StochasticReleaseTestProbe(name, hc);
   }
   return addedGroup;
}
