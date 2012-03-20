/*
 * pv.cpp
 *
 */

#include "../PetaVision/src/columns/buildandrun.hpp"
#include "GPUTestProbe.hpp"
#include "GPUTestForOnesProbe.hpp"
#include "GPUTestForTwosProbe.hpp"
#include "GPUTestForNinesProbe.hpp"

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
	if( !strcmp( keyword, "ArborTestProbe") ) {
		status = getLayerFunctionProbeParameters(groupname, keyword, hc, &targetlayer,
				&message, &filename);
		if (status != PV_SUCCESS) {
			fprintf(stderr, "Error reading params group \"%s\"\n", groupname);
			return addedProbe;
		}
		if( filename ) {
			addedProbe =  new GPUTestProbe(filename, targetlayer, message);
		}
		else {
			addedProbe =  new GPUTestProbe(targetlayer, message);
		}
		free(message); message=NULL; // message was alloc'ed in getLayerFunctionProbeParameters call
		if( !addedProbe ) {
			fprintf(stderr, "Group \"%s\": Unable to create %s\n", groupname, keyword);
		}
		assert(targetlayer);
		checknewobject((void *) addedProbe, keyword, groupname, hc);
		return addedProbe;
	}
	if( !strcmp( keyword, "ArborTestForOnesProbe") ) {
		status = getLayerFunctionProbeParameters(groupname, keyword, hc, &targetlayer,
				&message, &filename);
		if (status != PV_SUCCESS) {
			fprintf(stderr, "Error reading params group \"%s\"\n", groupname);
			return addedProbe;
		}
		if( filename ) {
			addedProbe =  new GPUTestForOnesProbe(filename, targetlayer, message);
		}
		else {
			addedProbe =  new GPUTestForOnesProbe(targetlayer, message);
		}
		free(message); message=NULL; // message was alloc'ed in getLayerFunctionProbeParameters call
		if( !addedProbe ) {
			fprintf(stderr, "Group \"%s\": Unable to create %s\n", groupname, keyword);
		}
		assert(targetlayer);
		checknewobject((void *) addedProbe, keyword, groupname, hc);
		return addedProbe;
	}
	if( !strcmp( keyword, "ArborTestForTwosProbe") ) {
		status = getLayerFunctionProbeParameters(groupname, keyword, hc, &targetlayer,
				&message, &filename);
		if (status != PV_SUCCESS) {
			fprintf(stderr, "Error reading params group \"%s\"\n", groupname);
			return addedProbe;
		}
		if( filename ) {
			addedProbe =  new GPUTestForTwosProbe(filename, targetlayer, message);
		}
		else {
			addedProbe =  new GPUTestForTwosProbe(targetlayer, message);
		}
		free(message); message=NULL; // message was alloc'ed in getLayerFunctionProbeParameters call
		if( !addedProbe ) {
			fprintf(stderr, "Group \"%s\": Unable to create %s\n", groupname, keyword);
		}
		assert(targetlayer);
		checknewobject((void *) addedProbe, keyword, groupname, hc);
		return addedProbe;
	}
	if( !strcmp( keyword, "ArborTestForNinesProbe") ) {
		status = getLayerFunctionProbeParameters(groupname, keyword, hc, &targetlayer,
				&message, &filename);
		if (status != PV_SUCCESS) {
			fprintf(stderr, "Error reading params group \"%s\"\n", groupname);
			return addedProbe;
		}
		if( filename ) {
			addedProbe =  new GPUTestForNinesProbe(filename, targetlayer, message);
		}
		else {
			addedProbe =  new GPUTestForNinesProbe(targetlayer, message);
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
