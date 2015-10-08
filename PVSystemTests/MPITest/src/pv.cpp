/*
 * pv.cpp
 *
 */

// using MPITestLayer
// activity/V are initialized to the global x/y/f position
// using uniform weights with total output strength of 1,
// all post synaptic cells should receive a total weighted input
// equal to thier global position
// MPITestProbe checks whether he above suppositions are satisfied

#include <columns/buildandrun.hpp>
#include <io/io.h>
#include "MPITestProbe.hpp"
#include "MPITestLayer.hpp"
#include <assert.h>

// use compiler directive in case MPITestLayer gets moved to PetaVision trunk
#define MAIN_USES_CUSTOMGROUP // TODO: rewrite using subclass of ParamGroupHandler

#ifdef MAIN_USES_CUSTOMGROUP
void * customgroup(const char * keyword, const char * name, HyPerCol * hc);
//int addcustom(HyPerCol * hc, int argc, char * argv[]);
// addcustom is for adding objects not supported by build().
#endif // MAIN_USES_ADDCUSTOM

int main(int argc, char * argv[]) {

   int status;
   PV_Init * initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   PV_Arguments * arguments = initObj->getArguments();
   if (arguments->getParamsFile()==NULL) {
      arguments->setParamsFile("input/MPI_test.params");
   }
#ifdef MAIN_USES_CUSTOMGROUP
   status = rebuildandrun(initObj, NULL, NULL, customgroup)==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
#else
   status = rebuildandrun(initObj);
#endif // MAIN_USES_ADDCUSTOM
   delete initObj;
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#ifdef MAIN_USES_CUSTOMGROUP

void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   HyPerLayer * targetLayer;
   void * addedGroup = NULL;
   char * msg = NULL;
   const char * filename;
   if( !strcmp(keyword, "MPITestLayer") ) {
      HyPerLayer * addedLayer = (HyPerLayer *) new MPITestLayer(name, hc);
      addedGroup = (void *) addedLayer;
   }
   else if( !strcmp( keyword, "MPITestProbe") ) {
      MPITestProbe * addedProbe = new MPITestProbe(name, hc);
      checknewobject((void *) addedProbe, keyword, name, hc);
      addedGroup = (void *) addedProbe;
   }
   int status = checknewobject(addedGroup, keyword, name, hc); // checknewobject tests addedObject against null, and either prints error message to stderr or success message to stdout.
   assert(status == PV_SUCCESS);
   return addedGroup;
}

#endif // MAIN_USES_CUSTOMGROUP
