/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include "CIFARGTLayer.hpp"
#include "HeliGTLayer.hpp"
#include "ConstGTLayer.hpp"
#include "ErrorMaskLayer.hpp"

#define MAIN_USES_CUSTOMGROUPS

#ifdef MAIN_USES_CUSTOMGROUPS
void * customgroup(const char * name, const char * groupname, HyPerCol * hc);
// customgroups is for adding objects not supported by build().
#endif // MAIN_USES_ADDCUSTOM

int main(int argc, char * argv[]) {

	int status;
#ifdef MAIN_USES_CUSTOMGROUPS
	status = buildandrun(argc, argv, NULL, NULL, &customgroup);
#else
	status = buildandrun(argc, argv);
#endif // MAIN_USES_CUSTOMGROUPS
	return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#ifdef MAIN_USES_CUSTOMGROUPS
void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   void * addedGroup = NULL;
   if ( !strcmp(keyword, "CIFARGTLayer") ) {
      addedGroup = new CIFARGTLayer(name, hc);
   }
   if ( !strcmp(keyword, "HeliGTLayer") ) {
      addedGroup = new HeliGTLayer(name, hc);
   }
   if ( !strcmp(keyword, "ConstGTLayer") ) {
      addedGroup = new ConstGTLayer(name, hc);
   }
   if ( !strcmp(keyword, "ErrorMaskLayer") ) {
      addedGroup = new ErrorMaskLayer(name, hc);
   }
   if (!addedGroup) {
      fprintf(stderr, "Group \"%s\": Unable to create %s\n", name, keyword);
      exit(EXIT_SUCCESS);
   }
   checknewobject((void *) addedGroup, keyword, name, hc);
   return addedGroup;
}
#endif
