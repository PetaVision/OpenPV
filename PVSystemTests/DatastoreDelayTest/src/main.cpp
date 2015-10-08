/*
 * DatastoreDelayTest.cpp
 *
 */

// using DatastoreDelayLayer, an input layer is filled with
// random data with the property that summing across four
// adjacent rows gives zeroes.
//
// On each timestep the data is rotated by one column.
// The input goes through four connections, with delays 0,1,2,3,
// each on the excitatory channel.
//
// The output layer should therefore be all zeros.

#include <columns/buildandrun.hpp>
#include <io/io.h>
#include "DatastoreDelayTestLayer.hpp"
#include "DatastoreDelayTestProbe.hpp"
#include <assert.h>

#define MAIN_USES_CUSTOMGROUP

#ifdef MAIN_USES_CUSTOMGROUP
void * customgroup(const char * keyword, const char * name, HyPerCol * hc);
#endif // MAIN_USES_CUSTOMGROUP

int main(int argc, char * argv[]) {

    int status;
#ifdef MAIN_USES_CUSTOMGROUP // TODO: rewrite using subclass of ParamGroupHandler
    PV_Init * initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
    PV_Arguments * arguments = initObj->getArguments();
    if (arguments->getParamsFile()==NULL) {
        arguments->setParamsFile("input/DatastoreDelayTest.params");
    }
    status = rebuildandrun(initObj, NULL, NULL, customgroup);
#else
    status = buildandrun(argc, argv);
#endif // MAIN_USES_ADDCUSTOM
    delete initObj;
    return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#ifdef MAIN_USES_CUSTOMGROUP

void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   PVParams * params = hc->parameters();
   void * addedGroup = NULL;
   if( !strcmp(keyword, "DatastoreDelayTestLayer") ) {
      HyPerLayer * addedLayer = new DatastoreDelayTestLayer(name, hc);
      addedGroup = (void *) addedLayer;
   }
   else if( !strcmp( keyword, "DatastoreDelayTestProbe" ) ) {
      DatastoreDelayTestProbe * addedProbe = new DatastoreDelayTestProbe(name, hc);
      addedGroup = (void *) addedProbe;
   }
   checknewobject((void *) addedGroup, keyword, name, hc);
   return addedGroup;
}

#endif // MAIN_USES_CUSTOMGROUP
