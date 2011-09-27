/*
 * pv.cpp
 *
 */

#include "../PetaVision/src/columns/buildandrun.hpp"
#include "ArborTestProbe.hpp"
#include "ArborTestForOnesProbe.hpp"

#define MAIN_USES_ADDCUSTOM

#ifdef MAIN_USES_ADDCUSTOM
int addcustom(HyPerCol * hc, int argc, char * argv[]);
// addcustom is for adding objects not supported by build().
#endif // MAIN_USES_ADDCUSTOM

int main(int argc, char * argv[]) {

    int status;
#ifdef MAIN_USES_ADDCUSTOM
    status = buildandrun(argc, argv, &addcustom);
#else
    status = buildandrun(argc, argv);
#endif // MAIN_USES_ADDCUSTOM
    return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#ifdef MAIN_USES_ADDCUSTOM
int addcustom(HyPerCol * hc, int argc, char * argv[]) {
	int status;
	PVParams * params = hc->parameters();
	int numGroups = params->numberOfGroups();
	for (int n = 0; n < numGroups; n++) {
		const char * kw = params->groupKeywordFromIndex(n);
		const char * name = params->groupNameFromIndex(n);
		HyPerLayer * targetlayer;
		const char * message;
		const char * filename;
		ArborTestProbe * addedProbe;
		ArborTestForOnesProbe * addedOnesProbe;
		if (!strcmp(kw, "ArborTestProbe")) {
			status = getLayerFunctionProbeParameters(name, kw, hc, &targetlayer,
					&message, &filename);
			if (status != PV_SUCCESS) {
				fprintf(stderr, "Skipping params group \"%s\"\n", name);
				continue;
			}
	         if( filename ) {
	            addedProbe =  new ArborTestProbe(filename, hc, message);
	         }
	         else {
	            addedProbe =  new ArborTestProbe(message);
	         }
	         if( !addedProbe ) {
	             fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
	         }
			assert(targetlayer);
			if( addedProbe ) targetlayer->insertProbe(addedProbe);
			checknewobject((void *) addedProbe, kw, name);
		}
		else if (!strcmp(kw, "ArborTestForOnesProbe")) {
			status = getLayerFunctionProbeParameters(name, kw, hc, &targetlayer,
					&message, &filename);
			if (status != PV_SUCCESS) {
				fprintf(stderr, "Skipping params group \"%s\"\n", name);
				continue;
			}
	         if( filename ) {
	        	 addedOnesProbe =  new ArborTestForOnesProbe(filename, hc, message);
	         }
	         else {
	        	 addedOnesProbe =  new ArborTestForOnesProbe(message);
	         }
	         if( !addedOnesProbe ) {
	             fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
	         }
			assert(targetlayer);
			if( addedOnesProbe ) targetlayer->insertProbe(addedOnesProbe);
			checknewobject((void *) addedOnesProbe, kw, name);
		}
	}
	return PV_SUCCESS;
}
#endif // MAIN_USES_ADDCUSTOM
