/*
 * pv.cpp
 *
 */

#include "../PetaVision/src/columns/buildandrun.hpp"
#include "TextStreamProbe.hpp"

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
    int status;
    LayerProbe * addedProbe = NULL;
    HyPerLayer * targetlayer;
    char * message = NULL;
    const char * filename;
    if( !strcmp( keyword, "TextStreamProbe") ) {
        status = getLayerFunctionProbeParameters(groupname, keyword, hc, &targetlayer,
                &message, &filename);
        if (status != PV_SUCCESS) {
            fprintf(stderr, "Error reading params group \"%s\"\n", groupname);
            return addedProbe;
        }
        int display_period = hc->parameters()->value(groupname, "displayPeriod", 1);
        if( filename ) {
            addedProbe =  new TextStreamProbe(filename, targetlayer, display_period);
        }
        else {
            addedProbe =  new TextStreamProbe(NULL, targetlayer, display_period);
        }
        free(message); message=NULL; // message was alloc'ed in getLayerFunctionProbeParameters call
        if( !addedProbe ) {
            fprintf(stderr, "Group \"%s\": Unable to create %s\n", groupname, keyword);
        }
        assert(targetlayer);
        checknewobject((void *) addedProbe, keyword, groupname, hc);
        return addedProbe;
    }
    assert(!addedProbe);
    fprintf(stderr, "Unrecognized params keyword \"%s\"\n", keyword);
    return addedProbe;
}
#endif // MAIN_USE_CUSTOMGROUPS
