/*
 * LateralConn.cpp
 *
 *  Created on: Nov 25, 2010
 *      Author: pschultz
 */

#include "LateralConn.hpp"

namespace PV {

PVConnParams defaultConnParamsLateral =
{
   /*delay*/ 0, /*fixDelay*/ 0, /*varDelayMin*/ 0, /*varDelayMax*/ 0, /*numDelay*/ 1,
   /*isGraded*/ 0, /*vel*/ 45.248, /*rmin*/ 0.0, /*rmax*/ 4.0
};

LateralConn::LateralConn(const char * name, HyPerCol *hc,
	    HyPerLayer * pre, HyPerLayer * post, int channel) {
    initialize_base();
    initialize(name, hc, pre, post, channel);
}  // end of LateralConn::LateralConn(const char *, HyPerCol *, HyPerLayer *, HyPerLayer *, int)

int LateralConn::initialize(const char * name, HyPerCol * hc,
	    HyPerLayer * pre, HyPerLayer * post, int channel) {

	PVLayer * preclayer = pre->getCLayer();
	PVLayer * postclayer = post->getCLayer();
	if( memcmp(&preclayer->loc,&postclayer->loc,sizeof(PVLayerLoc)) ) {
	    fprintf( stderr,
	             "LateralConn: %s and %s do not have the same dimensions\n",
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
    setParams(inputParams, &defaultConnParamsLateral);

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
}  // end of LateralConn::initialize(const char *, HyPerCol *, HyPerLayer *, HyPerLayer *, int)

int LateralConn::updateWeights(int axonID) {
    return EXIT_SUCCESS;
}  // end of LateralConn::updateWeights(int)

PVPatch ** LateralConn::initializeWeights(PVPatch ** patches, int numPatches,
          const char * filename) {
    // initialize to identity
	int xc = nxp/2;
	int yc = nyp/2;

	int numKernels = numDataPatches(0);
    for( int k=0; k < numKernels; k++ ) {
        PVPatch * kp = getKernelPatch(k);
        int idx = kIndex(xc, yc, k, nxp, nyp, nfp);
        kp->data[idx] = 1;
        // for( int l=0; l < kp->nf; l++ ) {
        //     kp->data[l] = l==k;
        // }
    }
    return patches;
}  // end of LateralConn::initializeWeights(PVPatch **, int)

}  // end of namespace PV block
