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

#include "../PetaVision/src/columns/buildandrun.hpp"
#include "../PetaVision/src/io/io.c"
#include "MPITestProbe.hpp"
#include "MPITestLayer.hpp"
#include <assert.h>

// use compiler directive in case MPITestLayer gets moved to PetaVision trunk
#define MAIN_USES_CUSTOMGROUP

#ifdef MAIN_USES_CUSTOMGROUP
void * customgroup(const char * keyword, const char * name, HyPerCol * hc);
//int addcustom(HyPerCol * hc, int argc, char * argv[]);
// addcustom is for adding objects not supported by build().
#endif // MAIN_USES_ADDCUSTOM

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
       cl_args[2] = strdup("input/MPI_test.params");
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
   HyPerLayer * targetLayer;
   void * addedGroup = NULL;
   const char * msg;
   const char * filename;
   if( !strcmp(keyword, "MPITestLayer") ) {
	   HyPerLayer * addedLayer = (HyPerLayer *) new MPITestLayer(name, hc);
      int status = checknewobject((void *) addedLayer, keyword, name, hc); // checknewobject tests addedObject against null, and either prints error message to stderr or success message to stdout.
      assert(status == PV_SUCCESS);
      addedGroup = (void *) addedLayer;
   }
   else if( !strcmp( keyword, "MPITestProbe") ) {
	   MPITestProbe * addedProbe;
      int status = getLayerFunctionProbeParameters(name, keyword, hc, &targetLayer, &msg, &filename);
      if( status == PV_SUCCESS ) {
         addedProbe = new MPITestProbe(filename, hc, msg);
      }
      if( addedProbe != NULL ) {
         assert(targetLayer);
         targetLayer->insertProbe(addedProbe);
      }
      addedGroup = (void *) addedProbe;
   }
   return addedGroup;
}

#endif // MAIN_USES_CUSTOMGROUP
