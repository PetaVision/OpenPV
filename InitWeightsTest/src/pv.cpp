/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include "HyPerConnDebugInitWeights.hpp"
#include "KernelConnDebugInitWeights.hpp"
#include "InitWeightTestProbe.hpp"

// customgroups is for adding objects not understood by build().
void * customgroups(const char * keyword, const char * name, HyPerCol * hc);

int main(int argc, char * argv[]) {
   return buildandrun(argc, argv, NULL, NULL, &customgroups)==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * customgroups(const char * keyword, const char * name, HyPerCol * hc) {
   bool errorFound = false;
   void * newobject = NULL;
   PVParams * params = hc->parameters();
   if (!strcmp( keyword, "HyPerConnDebugInitWeights") ) {
      newobject = new HyPerConnDebugInitWeights(name, hc);
   }
   else if (!strcmp(keyword, "KernelConnDebugInitWeights") ) {
      newobject = new KernelConnDebugInitWeights(name, hc);
   }
   else if (!strcmp(keyword, "InitWeightTestProbe")) {
      newobject = new InitWeightTestProbe(name, hc);
   }
   checknewobject(newobject, keyword, name, hc);
   return newobject;
}
