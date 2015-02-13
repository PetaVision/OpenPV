/*
 * PoolingConn.cpp
 *
 *  Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#include "PoolingConn.hpp"

namespace PV {
PoolingConn::PoolingConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) {
   initialize_base();
   initialize(name, hc, weightInitializer, weightNormalizer);
}  // end of PoolingConn::PoolingConn(const char *, HyPerCol *)

int PoolingConn::initialize_base() {
    pre2 = NULL;
    post2 = NULL;
    return PV_SUCCESS;
}

int PoolingConn::initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) {
   int status = HyPerConn::initialize(name, hc, weightInitializer, weightNormalizer);
   return status;
}

int PoolingConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_secondaryPreLayerName(ioFlag);
   ioParam_secondaryPostLayerName(ioFlag);
   return status;
}

void PoolingConn::ioParam_secondaryPreLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "secondaryPreLayerName", &preLayerName2);
}

void PoolingConn::ioParam_secondaryPostLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "secondaryPostLayerName", &postLayerName2);
}

int PoolingConn::communicateInitInfo() {
   int status = HyPerConn::communicateInitInfo();
   if (status != PV_SUCCESS) return status;
   pre2 = parent->getLayerFromName(preLayerName2);
   post2 = parent->getLayerFromName(postLayerName2);
   if (status == PV_SUCCESS) {
      status = checkLayersCompatible(pre, pre2) && checkLayersCompatible(post, post2) ? PV_SUCCESS : PV_FAILURE;
   }
   else {
      status = PV_FAILURE;
      exit(EXIT_FAILURE);
   }
   return status;
}

bool PoolingConn::checkLayersCompatible(HyPerLayer * layer1, HyPerLayer * layer2) {
	int nx1 = layer1->getLayerLoc()->nx;
	int nx2 = layer2->getLayerLoc()->nx;
	int ny1 = layer1->getLayerLoc()->ny;
	int ny2 = layer2->getLayerLoc()->ny;
	int nf1 = layer1->getLayerLoc()->nf;
	int nf2 = layer2->getLayerLoc()->nf;
    const PVHalo * halo1 = &layer1->getLayerLoc()->halo;
    const PVHalo * halo2 = &layer2->getLayerLoc()->halo;
    bool result = nx1==nx2 && ny1==ny2 && nf1==nf2 && halo1->lt==halo2->lt && halo1->rt==halo2->rt && halo1->dn==halo2->dn && halo1->up==halo2->up;
    if( !result ) {
    	const char * name1 = layer1->getName();
    	const char * name2 = layer2->getName();
        fprintf(stderr, "Group \"%s\": Layers \"%s\" and \"%s\" do not have compatible sizes\n", name, name1, name2);
        int len1 = (int) strlen(name1);
        int len2 = (int) strlen(name2);
        int len = len1 >= len2 ? len1 : len2;
        fprintf(stderr, "Layer \"%*s\": nx=%d, ny=%d, nf=%d, halo=(%d,%d,%d,%d)\n", len, name1, nx1, ny1, nf1, halo1->lt, halo1->rt, halo1->dn, halo1->up);
        fprintf(stderr, "Layer \"%*s\": nx=%d, ny=%d, nf=%d, halo=(%d,%d,%d,%d)\n", len, name2, nx2, ny2, nf2, halo2->lt, halo2->rt, halo2->dn, halo2->up);
    }
    return result;
}  // end of PoolingConn::PoolingConn(HyPerLayer *, HyPerLayer *)

int PoolingConn::updateWeights(int axonID) {
    int nPre = preSynapticLayer()->getNumNeurons();
    int nx = preSynapticLayer()->getLayerLoc()->nx;
    int ny = preSynapticLayer()->getLayerLoc()->ny;
    int nf = preSynapticLayer()->getLayerLoc()->nf;
    const PVHalo * halo = &preSynapticLayer()->getLayerLoc()->halo;
    for(int kPre=0; kPre<nPre;kPre++) {
        int kExt = kIndexExtended(kPre, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);

        size_t offset = getAPostOffset(kPre, axonID);
        pvdata_t preact = preSynapticLayer()->getCLayer()->activity->data[kExt];
        pvdata_t preact2 = getPre2()->getCLayer()->activity->data[kExt];
        PVPatch * weights = getWeights(kPre,axonID);
        int nyp = weights->ny;
        int nk = weights->nx * nfp;
        pvdata_t * postactRef = &(postSynapticLayer()->getCLayer()->activity->data[offset]);
        pvdata_t * postact2Ref = &(getPost2()->getCLayer()->activity->data[offset]);
        int sya = getPostNonextStrides()->sy;
        pvwdata_t * wtpatch = get_wData(axonID, kExt);
        int syw = syp;
        for( int y=0; y<nyp; y++ ) {
            int lineoffsetw = 0;
            int lineoffseta = 0;
            for( int k=0; k<nk; k++ ) {
                pvdata_t w = wtpatch[lineoffsetw + k] + dWMax*(preact*postactRef[lineoffseta + k]+preact2*postact2Ref[lineoffseta + k]);
                wtpatch[lineoffsetw + k] = w;
            }
            lineoffsetw += syw;
            lineoffseta += sya;
        }
    }
    lastUpdateTime = parent->simulationTime();

    return PV_SUCCESS;
}

PoolingConn::~PoolingConn() {
   free(preLayerName2); preLayerName2 = NULL;
   free(postLayerName2); postLayerName2 = NULL;
}

}  // end namespace PV
