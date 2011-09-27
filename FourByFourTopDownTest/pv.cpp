/*
 * pv.cpp
 *
 */

#include "../PetaVision/src/columns/buildandrun.hpp"
#include "../PetaVision/src/io/io.c"
#include "TopDownTestProbe.hpp"

void * customgroup(const char * keyword, const char * name, HyPerCol * hc);

int main(int argc, char * argv[]) {
   int paramfileabsent = pv_getopt_str(argc, argv, "-p", NULL);
   int num_cl_args;
   char ** cl_args;
   if( paramfileabsent ) {
      num_cl_args = argc + 2;
      cl_args = (char **) malloc(num_cl_args*sizeof(char *));
      cl_args[0] = argv[0];
      cl_args[1] = strdup("-p");
      cl_args[2] = strdup("input/FourByFourTopDownTest.params");
      for( int k=1; k<argc; k++) {
         cl_args[k+2] = strdup(argv[k]);
      }
   }
   else {
      num_cl_args = argc;
      cl_args = argv;
   }
   int status = buildandrun(num_cl_args, cl_args, NULL, NULL, customgroup)==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
   if( paramfileabsent ) {
      free(cl_args[1]);
      free(cl_args[2]);
      free(cl_args);
   }
   return status;
}

void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   HyPerLayer * targetLayer;
   void * addedGroup = NULL;
   const char * msg;
   const char * filename;
   if( !strcmp( keyword, "TopDownTestProbe") ) {
      TopDownTestProbe * addedProbe;
      int status = getLayerFunctionProbeParameters(name, keyword, hc, &targetLayer, &msg, &filename);
      if( status == PV_SUCCESS ) {
         float checkperiod = hc->parameters()->value(name, "checkPeriod", 0, true);
         addedProbe = new TopDownTestProbe(filename, hc, msg, checkperiod);
      }
      if( addedProbe != NULL ) {
         assert(targetLayer);
         targetLayer->insertProbe(addedProbe);
      }
      addedGroup = (void *) addedProbe;
   }
   return addedGroup;
}
