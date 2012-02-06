/*
 * PoolingGenConn.cpp
 *
 *  Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#include "PoolingGenConn.hpp"

namespace PV {

PoolingGenConn::PoolingGenConn(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, HyPerLayer * pre2, HyPerLayer * post2,
        ChannelType channel) {
        initialize_base();
        initialize(name, hc, pre, post, pre2, post2, channel, NULL, NULL);
}  // end of PoolingGenConn::PoolingGenConn(const char *, HyPerCol *,
   //   HyPerLayer *, HyPerLayer *, HyPerLayer *, HyPerLayer *, int)
PoolingGenConn::PoolingGenConn(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, HyPerLayer * pre2, HyPerLayer * post2,
        ChannelType channel, InitWeights *weightInit) {
        initialize_base();
        initialize(name, hc, pre, post, pre2, post2, channel, NULL, weightInit);
}  // end of PoolingGenConn::PoolingGenConn(const char *, HyPerCol *,
   //   HyPerLayer *, HyPerLayer *, HyPerLayer *, HyPerLayer *, int)

PoolingGenConn::PoolingGenConn(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, HyPerLayer * pre2, HyPerLayer * post2,
        ChannelType channel, const char * filename) {
        initialize_base();
        initialize(name, hc, pre, post, pre2, post2, channel, filename, NULL);
}  // end of PoolingGenConn::PoolingGenConn(const char *, HyPerCol *,
   //   HyPerLayer *, HyPerLayer *, HyPerLayer *, HyPerLayer *, int, const char *)
PoolingGenConn::PoolingGenConn(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, HyPerLayer * pre2, HyPerLayer * post2,
        ChannelType channel, const char * filename, InitWeights *weightInit) {
        initialize_base();
        initialize(name, hc, pre, post, pre2, post2, channel, filename, weightInit);
}  // end of PoolingGenConn::PoolingGenConn(const char *, HyPerCol *,
   //   HyPerLayer *, HyPerLayer *, HyPerLayer *, HyPerLayer *, int, const char *)

int PoolingGenConn::initialize_base() {
    pre2 = NULL;
    post2 = NULL;
    return PV_SUCCESS;
}

int PoolingGenConn::initialize(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, HyPerLayer * pre2, HyPerLayer * post2,
        ChannelType channel, const char * filename, InitWeights *weightInit) {
    GenerativeConn::initialize(name, hc, pre, post, channel, filename, weightInit);
    if( checkLayersCompatible(pre, pre2) && checkLayersCompatible(post, post2) ) {
        this->pre2 = pre2;
        this->post2 = post2;
        return PV_SUCCESS;
    }
    else {
        return PV_FAILURE;
    }
}  // end of PoolingGenConn::initialize(const char *, HyPerCol *,
   //   HyPerLayer *, HyPerLayer *, HyPerLayer *, HyPerLayer *, int, const char *)

int PoolingGenConn::initialize(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, HyPerLayer * pre2, HyPerLayer * post2,
        ChannelType channel) {
    return initialize(name, hc, pre, post, pre2, post2, channel, NULL, NULL);
}  // end of PoolingGenConn::initialize(const char *, HyPerCol *,
   //   HyPerLayer *, HyPerLayer *, HyPerLayer *, HyPerLayer *)

bool PoolingGenConn::checkLayersCompatible(HyPerLayer * layer1, HyPerLayer * layer2) {
	int nx1 = layer1->getLayerLoc()->nx;
	int nx2 = layer2->getLayerLoc()->nx;
	int ny1 = layer1->getLayerLoc()->ny;
	int ny2 = layer2->getLayerLoc()->ny;
	int nf1 = layer1->getLayerLoc()->nf;
	int nf2 = layer2->getLayerLoc()->nf;
	int nb1 = layer1->getLayerLoc()->nb;
	int nb2 = layer2->getLayerLoc()->nb;
    bool result = nx1==nx2 && ny1==ny2 && nf1==nf2 && nb1==nb2;
    if( !result ) {
    	const char * name1 = layer1->getName();
    	const char * name2 = layer2->getName();
        fprintf(stderr, "Group \"%s\": Layers \"%s\" and \"%s\" do not have compatible sizes\n", name, name1, name2);
        int len1 = (int) strlen(name1);
        int len2 = (int) strlen(name2);
        int len = len1 >= len2 ? len1 : len2;
        fprintf(stderr, "Layer \"%*s\": nx=%d, ny=%d, nf=%d, nb=%d\n", len, name1, nx1, ny1, nf1, nb1);
        fprintf(stderr, "Layer \"%*s\": nx=%d, ny=%d, nf=%d, nb=%d\n", len, name2, nx2, ny2, nf2, nb2);
    }
    return result;
}  // end of PoolingGenConn::PoolingGenConn(HyPerLayer *, HyPerLayer *)

int PoolingGenConn::updateWeights(int axonID) {
    int nPre = preSynapticLayer()->getNumNeurons();
    int nx = preSynapticLayer()->getLayerLoc()->nx;
    int ny = preSynapticLayer()->getLayerLoc()->ny;
    int nf = preSynapticLayer()->getLayerLoc()->nf;
    int pad = preSynapticLayer()->getLayerLoc()->nb;
    for(int kPre=0; kPre<nPre;kPre++) {
        int kExt = kIndexExtended(kPre, nx, ny, nf, pad);

        size_t offset = getAPostOffset(kPre, axonID);
        pvdata_t preact = preSynapticLayer()->getCLayer()->activity->data[kExt];
        pvdata_t preact2 = getPre2()->getCLayer()->activity->data[kExt];
        PVPatch * weights = getWeights(kPre,axonID);
        int nyp = weights->ny;
        int nk = weights->nx * weights->nf;
        pvdata_t * postactRef = &(postSynapticLayer()->getCLayer()->activity->data[offset]);
        pvdata_t * postact2Ref = &(getPost2()->getCLayer()->activity->data[offset]);
        int sya = getPostNonextStrides()->sy;
        pvdata_t * wtpatch = weights->data;
        int syw = weights->sy;
        for( int y=0; y<nyp; y++ ) {
            int lineoffsetw = 0;
            int lineoffseta = 0;
            for( int k=0; k<nk; k++ ) {
                float w = wtpatch[lineoffsetw + k] + relaxation*(preact*postactRef[lineoffseta + k]+preact2*postact2Ref[lineoffseta + k]);
                if( nonnegConstraintFlag && w < 0) w = 0;
                wtpatch[lineoffsetw + k] = w;
            }
            lineoffsetw += syw;
            lineoffseta += sya;
        }
    }
    // normalizeWeights now called in KernelConn::updateState // normalizeWeights( kernelPatches, numDataPatches(0) );
    lastUpdateTime = parent->simulationTime();

    return PV_SUCCESS;
}

}  // end namespace PV
