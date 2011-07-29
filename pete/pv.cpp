/*
 * pv.cpp
 *
 */

#include "../PetaVision/src/columns/buildandrun.hpp"
#include "ChannelProbe.hpp"

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
    for(int n=0; n<numGroups; n++) {
        const char * kw = params->groupKeywordFromIndex(n);
        const char * name = params->groupNameFromIndex(n);
        HyPerLayer * targetlayer;
        const char * message;
        const char * filename;
        if( !strcmp( kw, "ChannelProbe") ) {
            status = getLayerFunctionProbeParameters(name, kw, hc, &targetlayer, &message, &filename);
            if(status != PV_SUCCESS) {
                fprintf(stderr, "Skipping params group \"%s\"\n", name);
                continue;
            }
            if(filename == NULL) {
                fprintf(stderr, "ChannelProbe \"%s\": parameter filename must be set.  Skipping group.\n", name);
                continue;
            }
            ChannelType channelCode;
            int channelNo = params->value(name, "channelCode", -1);
            if( decodeChannel( channelNo, &channelCode ) != PV_SUCCESS) {
                fprintf(stderr, "%s \"%s\": parameter channelCode must be set.\n", kw, name);
                continue;
            }
            ChannelProbe * addedProbe = new ChannelProbe(filename, hc, channelCode);
            checknewobject((void *) addedProbe, kw, name);
        }
    }
    return PV_SUCCESS;
}
#endif // MAIN_USES_ADDCUSTOM
