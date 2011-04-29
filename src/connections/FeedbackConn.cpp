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

FeedbackConn::FeedbackConn(const char * name, HyPerCol *hc, ChannelType channel, GenerativeConn * ffconn) {
    initialize_base();
    initialize(name, hc, channel, NULL, ffconn);
}  // end of FeedbackConn::FeedbackConn(const char *, HyPerCol *, int, GenerativeConn *)

int FeedbackConn::initialize_base() {
    feedforwardConn = NULL;
    return PV_SUCCESS;
}

int FeedbackConn::initialize(const char * name, HyPerCol * hc,
            ChannelType channel, const char * filename, GenerativeConn * ffconn) {
    feedforwardConn = ffconn;
    GenerativeConn::initialize(name, hc, ffconn->postSynapticLayer(), ffconn->preSynapticLayer(), channel, filename);

    return PV_SUCCESS;
}  // end of FeedbackConn::initialize(const char *, HyPerCol *, HyPerLayer *, HyPerLayer *, int, const char * filename, GenerativeConn *)

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

PVPatch ** FeedbackConn::initializeWeights(PVPatch ** patches, int numPatches,
      const char * filename) {
    if( filename ) return KernelConn::initializeWeights(patches, numPatches, filename);

    transposeKernels();
    return patches;
}  // end of FeedbackConn::initializeWeights

int FeedbackConn::updateWeights(int axonID) {
    transposeKernels();
    return PV_SUCCESS;
}  // end of FeedbackConn::updateWeights(int);

int FeedbackConn::transposeKernels() {
    // compute the transpose of feedforwardConn->kernelPatches and
    // store into this->kernelPatches
    // assume scale factors are 1 and that nxp, nyp are odd.

    int xscalediff = pre->getXScale()-post->getXScale(); // scalediff>0 means feedbackconn's post--that is, the forward conn's pre--has a higher neuron density
    int yscalediff = pre->getYScale()-post->getYScale();
    int numFBKernelPatches = numDataPatches(0);
    int numFFKernelPatches = feedforwardConn->numDataPatches(0);

    if( xscalediff <= 0 && yscalediff <= 0) {
        int xscaleq = (int) powf(2,-xscalediff);
        int yscaleq = (int) powf(2,-yscalediff);

        for( int kernelnumberFB = 0; kernelnumberFB < numFBKernelPatches; kernelnumberFB++ ) {
            PVPatch * kpFB = getKernelPatch(kernelnumberFB);
            int nfFB = kpFB->nf;
               assert(numFFKernelPatches == nfFB);
            int nxFB = kpFB->nx;
            int nyFB = kpFB->ny;
            for( int kyFB = 0; kyFB < nyFB; kyFB++ ) {
                for( int kxFB = 0; kxFB < nxFB; kxFB++ ) {
                    for( int kfFB = 0; kfFB < nfFB; kfFB++ ) {
                        int kIndexFB = kIndex(kxFB,kyFB,kfFB,nxFB,nyFB,nfFB);
                        int kernelnumberFF = kfFB;
                        PVPatch * kpFF = feedforwardConn->getKernelPatch(kernelnumberFF);
                           assert(numFBKernelPatches == kpFF->nf * xscaleq * yscaleq);
                        int kfFF = featureIndex(kernelnumberFB, xscaleq, yscaleq, feedforwardConn->fPatchSize());
                        int kxFFoffset = kxPos(kernelnumberFB, xscaleq, yscaleq, feedforwardConn->fPatchSize());
                        int kxFF = (nxp - 1 - kxFB) * xscaleq + kxFFoffset;
                        int kyFFoffset = kyPos(kernelnumberFB, xscaleq, yscaleq, feedforwardConn->fPatchSize());
                        int kyFF = (nyp - 1 - kyFB) * yscaleq + kyFFoffset;
                        int kIndexFF = kIndex(kxFF, kyFF, kfFF, kpFF->nx, kpFF->ny, kpFF->nf);
                        // can the calls to kxPos, kyPos, featureIndex be replaced by one call to patchIndexToKernelIndex?
                        kpFB->data[kIndexFB] = kpFF->data[kIndexFF];
                    }
                }
            }
        }
    }
    else if( xscalediff > 0 && yscalediff > 0) {
        int xscaleq = (int) powf(2,xscalediff);
        int yscaleq = (int) powf(2,yscalediff);
        for( int kernelnumberFB = 0; kernelnumberFB < numFBKernelPatches; kernelnumberFB++ ) {
            PVPatch * kpFB = getKernelPatch(kernelnumberFB);
            int nxFB = kpFB->nx;
            int nyFB = kpFB->ny;
            int nfFB = kpFB->nf;
            for( int kyFB = 0; kyFB < nyFB; kyFB++ ) {
                int precelloffsety = kyFB % yscaleq;
                   for( int kxFB = 0; kxFB < nxFB; kxFB++ ) {
                    int precelloffsetx = kxFB % xscaleq;
                    for( int kfFB = 0; kfFB < nfFB; kfFB++ ) {
                        int kernelnumberFF = (precelloffsety*xscaleq + precelloffsetx)*nfFB + kfFB;
                        PVPatch * kpFF = feedforwardConn->getKernelPatch(kernelnumberFF);
                        int kxFF = (nxp-kxFB-1)/xscaleq;
                        assert(kxFF >= 0 && kxFF < feedforwardConn->xPatchSize());
                        int kyFF = (nyp-kyFB-1)/yscaleq;
                        assert(kyFF >= 0 && kyFF < feedforwardConn->yPatchSize());
                        int kfFF = kernelnumberFB;
                        assert(kfFF >= 0 && kfFF < feedforwardConn->fPatchSize());
                        int kIndexFF = kIndex(kxFF, kyFF, kfFF, kpFF->nx, kpFF->ny, kpFF->nf);
                        int kIndexFB = kIndex(kxFB, kyFB, kfFB, nxFB, nyFB, nfFB);
                        kpFB->data[kIndexFB] = kpFF->data[kIndexFF];
                    }
                }
            }
        }
    }
    else {
        fprintf(stderr,"xscalediff = %d, yscalediff = %d: this case not yet implemented.\n", xscalediff, yscalediff);
        exit(1);
    }

    return PV_SUCCESS;
}  // FeedbackConn::transposeKernels()

}  // end of namespace PV block

