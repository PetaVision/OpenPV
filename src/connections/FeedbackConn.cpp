/*
 * FeedbackConn.cpp
 *
 *  Created on: Nov 15, 2010
 *      Author: pschultz
 */

#include "FeedbackConn.hpp"

namespace PV {

FeedbackConn::FeedbackConn() {
    initialize_base();
}

FeedbackConn::FeedbackConn(const char * name, HyPerCol * hc, KernelConn * ffconn) :
        TransposeConn(name, hc, ffconn->postSynapticLayer(), ffconn->preSynapticLayer(), ffconn) {
    initialize_base();
    initialize(name, hc, ffconn);
}  // end of FeedbackConn::FeedbackConn(const char *, HyPerCol *, int, GenerativeConn *)

int FeedbackConn::initialize_base() {
   feedforwardConn = NULL;
   return PV_SUCCESS;
}

int FeedbackConn::initialize(const char * name, HyPerCol *hc, KernelConn * ffconn) {
   feedforwardConn = originalConn;
   //why doesn't feedbackconn call kernelconn's initialize???
   //kernelconns need this and the GPU stuff...
   initPatchToDataLUT();
   return PV_SUCCESS;
}

int FeedbackConn::setPatchSize(const char * filename) {
    int status = PV_SUCCESS;

    int xscaleDiff = pre->getXScale() - post->getXScale();
    // If feedforward conn is many-to-one, feedback conn is one-to-many.
    // Then xscaleDiff > 0.
    // Similarly, if feedforwardConn is one-to-many, xscaleDiff < 0.
    int yscaleDiff = pre->getYScale() - post->getYScale();

    nxp = feedforwardConn->xPatchSize();
    if(xscaleDiff > 0 ) {
        nxp *= (int) powf( 2, xscaleDiff );
    }
    else if(xscaleDiff < 0) {
        nxp /= (int) powf(2,-xscaleDiff);
        assert(feedforwardConn->xPatchSize()==nxp*powf( 2, (float) (-xscaleDiff) ));
    }
    nyp = feedforwardConn->yPatchSize();
    if(yscaleDiff > 0 ) {
        nyp *= (int) powf( 2, (float) yscaleDiff );
    }
    else if(yscaleDiff < 0) {
        nyp /= (int) powf(2,-yscaleDiff);
        assert(feedforwardConn->yPatchSize()==nyp*powf( 2, (float) (-yscaleDiff) ));
    }
    nfp = post->getLayerLoc()->nf;

    assert( checkPatchSize(nyp, pre->getXScale(), post->getXScale(), 'x') ==
            PV_SUCCESS );

    assert( checkPatchSize(nyp, pre->getYScale(), post->getYScale(), 'y') ==
            PV_SUCCESS );

    status = filename ? patchSizeFromFile(filename) : PV_SUCCESS;
    return status;
}  // end of FeedbackConn::setPatchSize(const char *)

PVPatch *** FeedbackConn::initializeWeights(PVPatch *** arbors, pvdata_t ** dataStart, int numPatches,
      const char * filename) {
    if( filename ) return KernelConn::initializeWeights(arbors, dataStart, numPatches, filename);

    transposeKernels();
    return arbors;
}  // end of FeedbackConn::initializeWeights

}  // end of namespace PV block

