/*
 * IdentConn.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#include "IdentConn.hpp"

namespace PV {

IdentConn::IdentConn() {
    initialize_base();
}

IdentConn::IdentConn(const char * name, HyPerCol *hc,
        HyPerLayer * pre, HyPerLayer * post, ChannelType channel) {
    initialize_base();
    initialize(name, hc, pre, post, channel);
}  // end of IdentConn::IdentConn(const char *, HyPerCol *, HyPerLayer *, HyPerLayer *, ChannelType)

int IdentConn::initialize_base() {
    // no IdentConn-specific data members to initialize
    return PV_SUCCESS;
}  // end of IdentConn::initialize_base()

int IdentConn::setPatchSize(const char * filename) {
    PVLayerLoc preLoc = pre->getCLayer()->loc;
    PVLayerLoc postLoc = post->getCLayer()->loc;
    int nfPre = pre->getLayerLoc()->nf;
    int nfPost = post->getLayerLoc()->nf;
    if( preLoc.nx != postLoc.nx || preLoc.ny != postLoc.ny || nfPre != nfPost ) {
        fprintf( stderr,
                 "IdentConn Error: %s and %s do not have the same dimensions\n",
                 pre->getName(),post->getName() );
        exit(1);
    }
    nxp = 1;
    nyp = 1;
    nfp = nfPre;

    return PV_SUCCESS;
}  // end of IdentConn::setPatchSize(const char *)

PVPatch ** IdentConn::initializeWeights(PVPatch ** patches, int numPatches,
          const char * filename) {
    int numKernels = numDataPatches(0);
    for( int k=0; k < numKernels; k++ ) {
        PVPatch * kp = getKernelPatch(k);
        assert(kp->nf == numKernels);
        for( int l=0; l < kp->nf; l++ ) {
            kp->data[l] = l==k;
        }
    }
    return patches;
}  // end of IdentConn::initializeWeights(PVPatch **, int, const char *)

}  // end of namespace PV block
