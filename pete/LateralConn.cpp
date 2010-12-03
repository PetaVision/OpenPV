/*
 * LateralConn.cpp
 *
 *  Created on: Nov 25, 2010
 *      Author: pschultz
 */

#include "LateralConn.hpp"

namespace PV {

LateralConn::LateralConn() {
    initialize_base();
}  // end of LateralConn::LateralConn()

LateralConn::LateralConn(const char * name, HyPerCol *hc,
        HyPerLayer * pre, HyPerLayer * post, int channel) {
    initialize_base();
    initialize(name, hc, pre, post, channel);
}  // end of LateralConn::LateralConn(const char *, HyPerCol *, HyPerLayer *, HyPerLayer *, int)

int LateralConn::initialize_base() {
    // no LateralConn-specific data members to initialize.
    return EXIT_SUCCESS;
}  // end of LateralConn::initialize_base()

int LateralConn::initialize(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, int channel) {
    return initialize(name, hc, pre, post, channel, NULL);
}

int LateralConn::initialize(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, int channel, const char * filename) {

    PVLayerLoc preLoc = pre->getCLayer()->loc;
    PVLayerLoc postLoc = post->getCLayer()->loc;
    if( preLoc.nx != postLoc.nx || preLoc.ny != postLoc.ny || preLoc.nBands != postLoc.nBands ) {
        fprintf( stderr,
                 "LateralConn Error: %s and %s do not have the same dimensions\n",
                 pre->getName(),post->getName() );
        exit(1);
    }
    GenerativeConn::initialize(name, hc, pre, post, channel, filename);

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
