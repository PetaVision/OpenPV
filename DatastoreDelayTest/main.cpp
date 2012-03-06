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

#include "../PetaVision/src/columns/buildandrun.hpp"
#include "../PetaVision/src/io/io.c"
#include "DatastoreDelayTestLayer.hpp"
#include "DatastoreDelayTestProbe.hpp"
#include <assert.h>

#define MAIN_USES_CUSTOMGROUP

#ifdef MAIN_USES_CUSTOMGROUP
void * customgroup(const char * keyword, const char * name, HyPerCol * hc);
#endif // MAIN_USES_CUSTOMGROUP

int main(int argc, char * argv[]) {

    int status;
#ifdef MAIN_USES_CUSTOMGROUP
    int paramfileabsent = pv_getopt_str(argc, argv, "-p", NULL);
    int num_cl_args;
    char ** cl_args;
    if( paramfileabsent ) {
       num_cl_args = argc + 2;
       cl_args = (char **) malloc(num_cl_args*sizeof(char *));
       cl_args[0] = argv[0];
       cl_args[1] = strdup("-p");
       cl_args[2] = strdup("input/DatastoreDelayTest.params");
       for( int k=1; k<argc; k++) {
          cl_args[k+2] = strdup(argv[k]);
       }
    }
    else {
       num_cl_args = argc;
       cl_args = argv;
    }
    status = buildandrun(num_cl_args, cl_args, NULL, NULL, customgroup)==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
    if( paramfileabsent ) {
       free(cl_args[1]);
       free(cl_args[2]);
       free(cl_args);
    }
#else
    status = buildandrun(argc, argv);
#endif // MAIN_USES_ADDCUSTOM
    return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#ifdef MAIN_USES_CUSTOMGROUP

void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   PVParams * params = hc->parameters();
   void * addedGroup = NULL;
   if( !strcmp(keyword, "DatastoreDelayTestLayer") ) {
      HyPerLayer * addedLayer = new DatastoreDelayTestLayer(name, hc);
      checknewobject((void *) addedLayer, keyword, name, hc);
      addedGroup = (void *) addedLayer;
   }
   else if( !strcmp( keyword, "DatastoreDelayTestProbe" ) ) {
      HyPerLayer * targetLayer;
      char * message = NULL;
      const char * filename;
      getLayerFunctionProbeParameters(name, keyword, hc, &targetLayer, &message, &filename);
      DatastoreDelayTestProbe * addedProbe = NULL;
      if( targetLayer ) {
         filename = params->stringValue(name, "probeOutputFile");
         addedProbe = new DatastoreDelayTestProbe(name, filename, hc, message);
         if( checknewobject((void *) addedProbe, keyword, name, hc) == PV_SUCCESS ) {
            targetLayer->insertProbe(addedProbe);
         }
      }
      else {
         fprintf(stderr, "Error: connection probe \"%s\" requires parameter \"targetConnection\".\n", name);
         addedProbe = NULL;
      }
      free(message); message=NULL; // message was alloc'ed in getLayerFunctionProbeParameters call
      addedGroup = (void *) addedProbe;
   }
   return addedGroup;
}

#endif // MAIN_USES_CUSTOMGROUP
