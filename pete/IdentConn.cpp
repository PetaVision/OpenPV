/*
 * IdentConn.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#include "IdentConn.hpp"

namespace PV {

PVConnParams defaultConnParamsIdent =
{
   /*delay*/ 0, /*fixDelay*/ 0, /*varDelayMin*/ 0, /*varDelayMax*/ 0, /*numDelay*/ 1,
   /*isGraded*/ 0, /*vel*/ 45.248, /*rmin*/ 0.0, /*rmax*/ 4.0
};

IdentConn::IdentConn(const char * name, HyPerCol *hc,
	    HyPerLayer * pre, HyPerLayer * post, int channel) {
    initialize_base();
    initialize(name, hc, pre, post, channel);
}  // end of IdentConn::IdentConn(const char *, HyPerCol *, int, GenLatConn *)

int IdentConn::initialize(const char * name, HyPerCol * hc,
	    HyPerLayer * pre, HyPerLayer * post, int channel) {

	PVLayer * preclayer = pre->getCLayer();
	PVLayer * postclayer = post->getCLayer();
	if( memcmp(&preclayer->loc,&postclayer->loc,sizeof(PVLayerLoc)) ) {
	    fprintf( stderr,
	             "IdentConn: %s and %s do not have the same dimensions\n",
	             pre->getName(),post->getName() );
	    return EXIT_FAILURE;
	}
    // lines below swiped from HyPerConn::initialize
    this->parent = hc;
    this->pre = pre;
    this->post = post;
    this->channel = channel;

    free(this->name);  // name will already have been set in initialize_base()
    this->name = strdup(name);
    assert(this->name != NULL);
    // lines above swiped from HyPerConn::initialize

    // HyPerConn::initialize(filename);
    const int arbor = 0;
    numAxonalArborLists = 1;

    assert(this->channel < postclayer->numPhis);

    this->connId = parent->numberOfConnections();

    PVParams * inputParams = parent->parameters();
    setParams(inputParams, &defaultConnParamsIdent);

    setPatchSize(NULL); // overridden

    // wPatches[arbor] = createWeights(wPatches[arbor]);
    wPatches[arbor] = createWeights(wPatches[arbor],
                          numWeightPatches(arbor),nxp,nyp,nfp); // don't need to override

    // initializeSTDP();

    // Create list of axonal arbors containing pointers to {phi,w,P,M} patches.
    //  weight patches may shrink
    // readWeights() should expect shrunken patches
    // initializeWeights() must be aware that patches may not be uniform
    createAxonalArbors();

    initializeWeights(wPatches[arbor], numWeightPatches(arbor), NULL);  // need to override
    assert(wPatches[arbor] != NULL);

    writeTime = parent->simulationTime();
    writeStep = inputParams->value(name, "writeStep", parent->getDeltaTime());

    parent->addConnection(this);
    // HyPerConn::initialize(filename);

    return EXIT_SUCCESS;
}  // end of IdentConn::initialize(const char *, HyPerCol *, HyPerLayer *, HyPerLayer *, int, const char * filename, GenLatConn *)

int IdentConn::setPatchSize(const char * filename) {
	int status = EXIT_SUCCESS;
    nxp = 1;
    nyp = 1;
    nfp = pre->getCLayer()->numFeatures;

    return status;
}  // end of IdentConn::setPatchSize(const char *)

PVPatch ** IdentConn::initializeWeights(PVPatch ** patches, int numPatches,
          const char * filename) {
	int numKernels = numDataPatches(0);
    for( int k=0; k < numKernels; k++ ) {
        PVPatch * kp = getKernelPatch(k);
        for( int l=0; l < kp->nf; l++ ) {
            kp->data[l] = l==k;
        }
    }
    return patches;
}

}  // end of namespace PV block
