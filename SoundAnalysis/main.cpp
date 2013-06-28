/*
 * pv.cpp
 *
 */

#include "../PetaVision/src/columns/buildandrun.hpp"

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
}
#endif // MAIN_USE_CUSTOMGROUPS
