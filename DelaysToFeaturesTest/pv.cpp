/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include "DelayTestProbe.hpp"

void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc);

int main(int argc, char * argv[]) {
	int status = buildandrun(argc, argv, NULL, NULL, &addcustomgroup);
	return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc) {
	int status;
	LayerProbe * addedProbe = NULL;
	HyPerLayer * targetlayer;
	char * message = NULL;
	const char * filename;
	if( !strcmp( keyword, "DelayTestProbe") ) {
		status = getLayerFunctionProbeParameters(groupname, keyword, hc, &targetlayer,
				&message, &filename);
		if (status != PV_SUCCESS) {
			fprintf(stderr, "Error reading params group \"%s\"\n", groupname);
			return addedProbe;
		}
		if( filename ) {
			addedProbe =  new DelayTestProbe(filename, targetlayer, message);
		}
		else {
			addedProbe =  new DelayTestProbe(targetlayer, message);
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
