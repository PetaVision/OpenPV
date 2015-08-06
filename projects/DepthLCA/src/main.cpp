/*
 * pv.cpp
 *
 */

#include "columns/buildandrun.hpp"
#include "LocalKernelConn.hpp"
//#include "BatchConn.hpp"
#include "SLPError.hpp"
#include "BinaryThresh.hpp"
//#include "DisparityMovie.hpp"
//#include "DisparityLCALayer.hpp"
//#include "SUPointProbe.hpp"

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

   if ( !strcmp(keyword, "LocalKernelConn") ) {
      addedGroup = new LocalKernelConn(name, hc);
   }
   //if ( !strcmp(keyword, "BatchConn") ) {
   //   addedGroup = new BatchConn(name, hc);
   //}
   if ( !strcmp(keyword, "SLPError") ) {
      addedGroup = new SLPError(name, hc);
   }
   if ( !strcmp(keyword, "BinaryThresh") ) {
      addedGroup = new BinaryThresh(name, hc);
   }
   //if ( !strcmp(keyword, "DisparityMovie") ) {
   //   addedGroup = new DisparityMovie(name, hc);
   //}
   //if ( !strcmp(keyword, "DisparityLCALayer") ) {
   //   addedGroup = new DisparityLCALayer(name, hc);
   //}
   //if ( !strcmp(keyword, "SUPointProbe") ) {
   //   addedGroup = new SUPointProbe(name, hc);
   //}
   return addedGroup;
}
#endif
