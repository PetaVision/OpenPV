/*
 * Main file for CloneKernelConnTest
 * To run, use arguments -p input/CloneKernelConnTest.params
 *
 */


#include "../PetaVision/src/columns/buildandrun.hpp"
#include "CloneKernelConnTestProbe.hpp"

#define MAIN_USES_CUSTOMGROUP

#ifdef MAIN_USES_CUSTOMGROUP
void * customgroup(const char * name, const char * groupname, HyPerCol * hc);
// customgroups is for adding objects not supported by build().
#endif // MAIN_USES_ADDCUSTOM

int main(int argc, char * argv[]) {

   int status;
#ifdef MAIN_USES_CUSTOMGROUP
   status = buildandrun(argc, argv, NULL, NULL, &customgroup);
#else
   status = buildandrun(argc, argv);
#endif // MAIN_USES_CUSTOMGROUP
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#ifdef MAIN_USES_CUSTOMGROUP
void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
	   int status;
	   // PVParams * params = hc->parameters();
	   // HyPerLayer * preLayer;
	   // HyPerLayer * postLayer;
	   LayerProbe * addedProbe;
	   void * addedGroup = NULL;
	   const char * filename;
	   HyPerLayer * targetlayer;
	   char * message = NULL;
	   bool errorFound;
	   if( !strcmp(keyword, "CloneKernelConnTestProbe") ) {
	      status = getLayerFunctionProbeParameters(name, keyword, hc, &targetlayer, &message, &filename);
	      errorFound = status!=PV_SUCCESS;
	      if( !errorFound ) {
	         PVBufType buf_type = BufV;
	         if (targetlayer->getSpikingFlag()) {
	            buf_type = BufActivity;
	         }
	         if( filename ) {
	            addedProbe = (LayerProbe *) new CloneKernelConnTestProbe(filename, targetlayer, message);
	         }
	         else {
	            addedProbe = (LayerProbe *) new CloneKernelConnTestProbe(targetlayer, message);
	         }
	         if( !addedProbe ) {
	            fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
	            errorFound = true;
	         }
	         if( !errorFound ) addedGroup = (void *) addedProbe;
	      }
	      free(message); message = NULL;
	   }
	   return addedGroup;
}
#endif // MAIN_USES_CUSTOMGROUP
