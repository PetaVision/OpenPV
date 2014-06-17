/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include "SoundProbe.hpp"
#include "CochlearLayer.hpp"
#include "inverseCochlearLayer.hpp"

#define MAIN_USES_CUSTOMGROUPS

#ifdef MAIN_USES_CUSTOMGROUPS
void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc);
// customgroups is for adding objects not supported by build().
#endif // MAIN_USES_ADDCUSTOM

int main(int argc, char * argv[]) {

	int status;
#ifdef MAIN_USES_CUSTOMGROUPS
	status = buildandrun(argc, argv, NULL, NULL, &addcustomgroup);
#else
	status = buildandrun(argc, argv);
#endif // MAIN_USES_CUSTOMGROUPS
	return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#ifdef MAIN_USES_CUSTOMGROUPS
void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc) {
    void * addedGroup = NULL;
    if ( !strcmp(keyword, "SoundProbe") ) {
       addedGroup = new SoundProbe(groupname, hc);
    }
    if ( !strcmp(keyword, "CochlearLayer") ) {
       addedGroup = new CochlearLayer(groupname, hc);
    }
    if ( !strcmp(keyword, "inverseCochlearLayer") ) {
       addedGroup = new inverseCochlearLayer(groupname, hc);
    }
    return addedGroup;
}
#endif // MAIN_USE_CUSTOMGROUPS
