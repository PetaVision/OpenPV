/*
 * Main file for CloneKernelConnTest
 * To run, use arguments -p input/CloneKernelConnTest.params
 *
 */


#include <columns/buildandrun.hpp>
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
          addedProbe = (LayerProbe *) new CloneKernelConnTestProbe(name, hc);
	   }
	   return addedGroup;
}
#endif // MAIN_USES_CUSTOMGROUP
