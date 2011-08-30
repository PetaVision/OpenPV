/*
 * pv.cpp
 *
 */

#include "../PetaVision/src/columns/buildandrun.hpp"
#include "MPITestProbe.hpp"

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
		MPITestProbe * addedProbe;
		if (!strcmp(kw, "MPITestProbe")) {
			status = getLayerFunctionProbeParameters(name, kw, hc, &targetlayer,
					&message, &filename);
			if (status != PV_SUCCESS) {
				fprintf(stderr, "Skipping params group \"%s\"\n", name);
				continue;
			}
			PVBufType buf_type = BufV;
	         if( filename ) {
	            addedProbe =  new MPITestProbe(filename, hc, buf_type, message);
	         }
	         else {
	            addedProbe =  new MPITestProbe(buf_type, message);
	         }
	         if( !addedProbe ) {
	             fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
	         }
			assert(targetlayer);
			if( addedProbe ) targetlayer->insertProbe(addedProbe);
			checknewobject((void *) addedProbe, kw, name);
		}
	}
	return PV_SUCCESS;
}
#endif // MAIN_USES_ADDCUSTOM
