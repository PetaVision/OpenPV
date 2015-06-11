/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include "CustomGroupHandler.hpp"
#include <MLearningGroupHandler.hpp>

int main(int argc, char * argv[]) {

	int status;
   PV::ParamGroupHandler * customGroupHandlers[2];
   customGroupHandlers[0] = new PVMLearning::MLearningGroupHandler;
   customGroupHandlers[1] = new PV::CustomGroupHandler;

	status = buildandrun(argc, argv, NULL, NULL, customGroupHandlers, 2);
	return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

//#ifdef MAIN_USES_CUSTOMGROUPS
//void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
//   void * addedGroup = NULL;
//   if( !strcmp(keyword, "CIFARGTLayer") ) {
//      addedGroup= new CIFARGTLayer(name, hc);
//   }
//   if( !strcmp(keyword, "ProbeLayer") ) {
//      addedGroup= new ProbeLayer(name, hc);
//   }
//   if( !addedGroup) {
//      fprintf(stderr, "Group \"%s\": Unable to create layer\n", name);
//      exit(-1);
//   }
//   return addedGroup;
//}
//#endif
