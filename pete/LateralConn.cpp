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
        HyPerLayer * pre, HyPerLayer * post, ChannelType channel) {
    initialize_base();
    initialize(name, hc, pre, post, channel);
}  // end of LateralConn::LateralConn(const char *, HyPerCol *, HyPerLayer *, HyPerLayer *, ChannelType)

int LateralConn::initialize_base() {
    // no LateralConn-specific data members to initialize.
    return EXIT_SUCCESS;
}  // end of LateralConn::initialize_base()

int LateralConn::initialize(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, ChannelType channel) {
    return initialize(name, hc, pre, post, channel, NULL);
}

int LateralConn::initialize(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename) {

    const PVLayerLoc * preLoc = pre->getLayerLoc();
    const PVLayerLoc * postLoc = post->getLayerLoc();
    if( preLoc->nx != postLoc->nx || preLoc->ny != postLoc->ny ||
            preLoc->nf != postLoc->nf ) {
        fprintf( stderr,
                 "LateralConn Error: %s and %s do not have the same dimensions\n",
                 pre->getName(),post->getName() );
        exit(1);
    }
    GenerativeConn::initialize(name, hc, pre, post, channel, filename);
	int prePad = pre->getLayerLoc()->nb;
	int xPatchHead = zPatchHead(0, nxp, 0, 0);
	int yPatchHead = zPatchHead(0, nyp, 0, 0);
	if( -xPatchHead > prePad || -yPatchHead > prePad) {
	   int needed = (-xPatchHead > -yPatchHead ) ? -xPatchHead : -yPatchHead;
	   fprintf(stderr, "Connection \"%s\": Layer \"%s\" has marginWidth %d; connection requires at least %d.\n",name, pre->getName(), prePad, needed);
	   abort();
	}
	// If -xPatchHead > prePad || -yPatchHead > prePad evaluates to true, updateWeights will crash.
	// This should be fixed.

    return EXIT_SUCCESS;
}  // end of LateralConn::initialize(const char *, HyPerCol *, HyPerLayer *, HyPerLayer *, ChannelType)

int LateralConn::updateWeights(int axonID) {
	const PVLayerLoc * preLoc = pre->getLayerLoc();
	int nxPre = preLoc->nx;
	int nyPre = preLoc->ny;
	int nfPre = preLoc->nf;
	const PVLayerLoc * postLoc = post->getLayerLoc();
	int nxPost = postLoc->nx;
	int nyPost = postLoc->ny;
	int nfPost = postLoc->nf;
	assert(nxPre == nxPost && nyPre == nyPost && nfPre == nfPost);
	int prePad = preLoc->nb;
	int nxExt = nxPre + 2*prePad;
	int nyExt = nyPre + 2*prePad;
	int xPatchHead = zPatchHead(0, nxp, 0, 0);
	int yPatchHead = zPatchHead(0, nyp, 0, 0);
    // Fix so that the code below won't break if the presynaptic marginWidth is insufficient.
	// Should go through axonalArbor() to handle shrunken patches correctly.
	// When fixed, can remove if( -xPatchHead > prePad || -yPatchHead > prePad) statement at end
	// of initialize()

	const pvdata_t * preExt = pre->getLayerData();
    for( int kRestr = 0; kRestr < post->getNumNeurons(); kRestr++ ) {
        if( pvdata_t apost = post->getV()[kRestr] ) {
        	int xRestr = kxPos(kRestr, nxPost, nyPost, nfPost);
        	int yRestr = kyPos(kRestr, nxPost, nyPost, nfPost);
        	// int feature = featureIndex(kRestr, postnx, postny, postnf);
            for( int v=0; v<nyp; v++ ) {
                for( int u=0; u<nxp; u++) {
                	if( u != v ) {
                        for( int p=0; p<nfp; p++ ) {
                            pvdata_t * kpdata = getKernelPatch(p)->data;
                            int kIndex1 = kIndex(xRestr - (u + xPatchHead) + prePad, yRestr - (v + yPatchHead) + prePad, p, nxExt, nyExt, nfPre);
                            assert(kIndex1 >= 0 && kIndex1 < pre->getNumExtended());
                            int kIndex2 = kIndex(xRestr - (v + yPatchHead) + prePad, yRestr - (u + xPatchHead) + prePad, p, nxExt, nyExt, nfPre);
                            assert(kIndex2 >= 0 && kIndex2 < pre->getNumExtended());
                            int kIndexPatch = kIndex(u, v, p, nxp, nyp, nfp);
                            kpdata[kIndexPatch] -= relaxation*apost*(preExt[kIndex1]+preExt[kIndex2]);
                        }
                	}
                }
            }
        }
    }
    return EXIT_SUCCESS;
}  // end of LateralConn::updateWeights(int)

PVPatch ** LateralConn::initializeWeights(PVPatch ** patches, int numPatches,
          const char * filename) {
    // initialize to identity
	if( nxp != nyp ) {
	    fprintf(stderr, "Lateral connection \"%s\":\n", name);
	    fprintf(stderr, "nxp must equal nyp since lateral connections are symmetric\n");
	    exit(1);
	}
    int xc = - zPatchHead(0, nxp, 0, 0); // zPatchHead(0,nxp,0,0) is the index of the leftmost
    int yc = - zPatchHead(0, nyp, 0, 0); // pixel in the patch; its negative is the index of the center

    int numKernels = numDataPatches(0);
    for( int k=0; k < numKernels; k++ ) {
        PVPatch * kp = getKernelPatch(k);
        int idx = kIndex(xc, yc, k, nxp, nyp, nfp);
        kp->data[idx] = 1;
    }
    return patches;
}  // end of LateralConn::initializeWeights(PVPatch **, int)

}  // end of namespace PV block
