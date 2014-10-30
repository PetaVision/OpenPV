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
#include "PlasticConnTestLayer.hpp"
#include "PlasticTestConn.hpp"
#include "PlasticConnTestProbe.hpp"
#include <connections/HyPerConn.hpp>
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
       cl_args = (char **) malloc(num_cl_args*sizeof(char *)+1);
       cl_args[0] = argv[0];
       cl_args[1] = strdup("-p");
       cl_args[2] = strdup("input/PlasticConnTest.params");
       for( int k=1; k<argc; k++) {
          cl_args[k+2] = strdup(argv[k]);
       }
       cl_args[argc+2] = NULL;
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
   char * preLayerName = NULL;
   char * postLayerName = NULL;
   void * addedGroup = NULL;
   const char * filename;
   if( !strcmp(keyword, "PlasticConnTestLayer") ) {
      HyPerLayer * addedLayer = new PlasticConnTestLayer(name, hc);
      checknewobject((void *) addedLayer, keyword, name, hc);
      addedGroup = (void *) addedLayer;
   }
   else if( !strcmp(keyword, "PlasticTestConn") ) {
      HyPerConn * addedConn = (HyPerConn * ) new PlasticTestConn(name, hc);
      checknewobject((void *) addedConn, keyword, name, hc);
      addedGroup = (void *) addedConn;
   }
   else if( !strcmp( keyword, "PlasticConnTestProbe" ) ) {
      PlasticConnTestProbe * addedProbe = NULL;
      const char * targetConnName = params->stringValue(name, "targetConnection");
      BaseConnection * targetBaseConn = hc->getConnFromName(targetConnName);
      HyPerConn * targetHyPerConn = dynamic_cast<HyPerConn *>(targetBaseConn);
      if (targetHyPerConn) {
         addedProbe = new PlasticConnTestProbe(name, hc);
         if( checknewobject((void *) addedProbe, keyword, name, hc) == PV_SUCCESS ) {
         }
      }
      else {
         fprintf(stderr, "Error: connection probe \"%s\" requires parameter \"targetConnection\".\n", name);
         addedProbe = NULL;
      }
      addedGroup = (void *) addedProbe;
   }
   free(preLayerName);
   free(postLayerName);
   return addedGroup;
}

#endif // MAIN_USES_CUSTOMGROUP
