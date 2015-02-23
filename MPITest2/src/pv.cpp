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
#include <io/io.c>
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
   // If params file was not specified, add input/MPI_test.params to command line arguments
   int paramfileabsent = pv_getopt_str(argc, argv, "-p", NULL/*sVal*/, NULL/*paramusage*/);
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
#ifdef MAIN_USES_CUSTOMGROUP
   status = buildandrun(num_cl_args, cl_args, NULL, NULL, customgroup)==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
#else
   status = buildandrun(argc, argv);
#endif // MAIN_USES_ADDCUSTOM
   if( paramfileabsent ) {
      free(cl_args[1]);
      free(cl_args[2]);
      free(cl_args);
   }
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
