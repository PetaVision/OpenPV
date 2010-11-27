/*
 * FeedbackConn.cpp
 *
 *  Created on: Nov 15, 2010
 *      Author: pschultz
 */

#include "FeedbackConn.hpp"

namespace PV {

PVConnParams defaultConnParamsFB =
{
   /*delay*/ 0, /*fixDelay*/ 0, /*varDelayMin*/ 0, /*varDelayMax*/ 0, /*numDelay*/ 1,
   /*isGraded*/ 0, /*vel*/ 45.248, /*rmin*/ 0.0, /*rmax*/ 4.0
};

FeedbackConn::FeedbackConn(const char * name, HyPerCol *hc, int channel, GenerativeConn * ffconn) {
    initialize_base();
    initialize(name, hc, channel, NULL, ffconn);
}  // end of FeedbackConn::FeedbackConn(const char *, HyPerCol *, int, GenerativeConn *)

int FeedbackConn::initialize(const char * name, HyPerCol * hc,
            int channel, const char * filename, GenerativeConn * ffconn) {
    feedforwardConn = ffconn;

    // lines below swiped from HyPerConn::initialize
    this->parent = hc;
    this->pre = ffconn->postSynapticLayer();
    this->post = ffconn->preSynapticLayer();
    this->channel = channel;

    free(this->name);  // name will already have been set in initialize_base()
    this->name = strdup(name);
    assert(this->name != NULL);
    // lines above swiped from HyPerConn::initialize

    // HyPerConn::initialize(filename);
    const int arbor = 0;
    numAxonalArborLists = 1;

    assert(this->channel < post->clayer->numPhis);

    this->connId = parent->numberOfConnections();

    PVParams * inputParams = parent->parameters();
    setParams(inputParams, &defaultConnParamsFB);

    setPatchSize(filename); // overridden

    // wPatches[arbor] = createWeights(wPatches[arbor]);
    wPatches[arbor] = createWeights(wPatches[arbor],
                          numWeightPatches(arbor),nxp,nyp,nfp); // don't need to override

    // initializeSTDP();

    // Create list of axonal arbors containing pointers to {phi,w,P,M} patches.
    //  weight patches may shrink
    // readWeights() should expect shrunken patches
    // initializeWeights() must be aware that patches may not be uniform
    createAxonalArbors();

    initializeWeights(wPatches[arbor], numWeightPatches(arbor), filename);  // need to override
    assert(wPatches[arbor] != NULL);

    writeTime = parent->simulationTime();
    writeStep = inputParams->value(name, "writeStep", parent->getDeltaTime());

    parent->addConnection(this);
    // HyPerConn::initialize(filename);

    weightUpdatePeriod = ffconn->getWeightUpdatePeriod();
    nextWeightUpdate = weightUpdatePeriod;
    relaxation = ffconn->getRelaxation();
    return EXIT_SUCCESS;
}  // end of FeedbackConn::initialize(const char *, HyPerCol *, HyPerLayer *, HyPerLayer *, int, const char * filename, GenerativeConn *)

int FeedbackConn::setPatchSize(const char * filename) {
	int status = EXIT_SUCCESS;
    nxp = feedforwardConn->xPatchSize();
    nyp = feedforwardConn->yPatchSize();
    nfp = post->getCLayer()->numFeatures;

    if (filename != NULL) { // copied from HyPerConn::setPatchSize.  Would be better to modularize out the nxp, nyp, nfp calculations
       int filetype, datatype;
       double time = 0.0;
       const PVLayerLoc loc = this->pre->clayer->loc;

       int wgtParams[NUM_WGT_PARAMS];
       int numWgtParams = NUM_WGT_PARAMS;

       Communicator * comm = parent->icCommunicator();

       status = pvp_read_header(filename, comm, &time, &filetype, &datatype, wgtParams, &numWgtParams);
       if (status < 0) return status;

       status = checkPVPFileHeader(&loc, wgtParams, numWgtParams);
       if (status < 0) return status;

       // reconcile differences with inputParams
       status = checkWeightsHeader(filename, wgtParams);
    }
    return status;
}  // end of FeedbackConn::setPatchSize(const char *)

PVPatch ** FeedbackConn::initializeWeights(PVPatch ** patches, int numPatches,
      const char * filename) {
    if( filename ) return KernelConn::initializeWeights(patches, numPatches, filename);

    transposeKernels();
    return patches;
}  // end of FeedbackConn::initializeWeights

int FeedbackConn::updateWeights(int axonID) {
	printf("updateWeights %s\n", name);
	transposeKernels();
    return EXIT_SUCCESS;
}  // end of FeedbackConn::updateWeights(int);

int FeedbackConn::transposeKernels() {
	// compute the transpose of feedforwardConn->kernelPatches and
	// store into this->kernelPatches
    // assume scale factors are 1 and that nxp, nyp are odd.
    int numFFKernelPatches = feedforwardConn->numDataPatches(0);
    int numFBKernelPatches = numDataPatches(0);
    for( int kernelnumberFB=0; kernelnumberFB < numFBKernelPatches; kernelnumberFB++ ) {
        PVPatch * kpFB = getKernelPatch(kernelnumberFB); //patches[0]?
        int nxFB = kpFB->nx;
        int nyFB = kpFB->ny;
        int nfFB = kpFB->nf;
        assert(numFFKernelPatches == nfFB);
        for( int kyFB = 0; kyFB < nyFB; kyFB++ ) {
            int kyFF = nyFB-1-kyFB;
            for( int kxFB = 0; kxFB < nxFB; kxFB++ ) {
                int kxFF = nxFB-1-kxFB;
                for( int kfFB = 0; kfFB < nfFB; kfFB++ ) {
                    int kfFF = kernelnumberFB;
                    int kernelnumberFF = kfFB;
                    PVPatch * kpFF = feedforwardConn->getKernelPatch(kernelnumberFF);
                    int kIndexFF = kIndex(kxFF,kyFF,kfFF,kpFF->nx,kpFF->ny,kpFF->nf);
                    int kIndexFB = kIndex(kxFB,kyFB,kfFB,nxFB,nyFB,nfFB);
                    kpFB->data[kIndexFB] = kpFF->data[kIndexFF];
                }
            }
        }
    }
    return EXIT_SUCCESS;
}  // FeedbackConn::transposeKernels()

}  // end of namespace PV block

