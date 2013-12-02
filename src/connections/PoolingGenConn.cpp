/*
 * PoolingGenConn.cpp
 *
 *  Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#include "PoolingGenConn.hpp"

namespace PV {
PoolingGenConn::PoolingGenConn(const char * name, HyPerCol * hc,
      const char * pre_layer_name, const char * post_layer_name,
      const char * pre_layer_name2, const char * post_layer_name2,
      const char * filename, InitWeights *weightInit) {
   initialize_base();
   initialize(name, hc, pre_layer_name, post_layer_name, pre_layer_name2, post_layer_name2, filename, weightInit);
}  // end of PoolingGenConn::PoolingGenConn(const char *, HyPerCol *,
   //   HyPerLayer *, HyPerLayer *, HyPerLayer *, HyPerLayer *, int, const char *)

int PoolingGenConn::initialize_base() {
    pre2 = NULL;
    post2 = NULL;
    return PV_SUCCESS;
}

int PoolingGenConn::initialize(const char * name, HyPerCol * hc,
      const char * pre_layer_name, const char * post_layer_name,
      const char * pre_layer_name2, const char * post_layer_name2,
      const char * filename, InitWeights *weightInit) {
   int status;
   // PVParams * params = hc->parameters();
   preLayerName2 = strdup(pre_layer_name2);
   postLayerName2 = strdup(post_layer_name2);
   status = GenerativeConn::initialize(name, hc, pre_layer_name, post_layer_name, filename, weightInit);
   return status;
}  // end of PoolingGenConn::initialize(const char *, HyPerCol *,
   //   HyPerLayer *, HyPerLayer *, HyPerLayer *, HyPerLayer *, int, const char *, InitWeights *)

int PoolingGenConn::setParams(PVParams * params) {
   int status = GenerativeConn::setParams(params);
   readSlownessFlag(params);
   if (status == PV_SUCCESS) status = readSlownessPre(params);
   if (status == PV_SUCCESS) status = readSlownessPost(params);
   if (status != PV_SUCCESS) {
      abort();
   }
   return status;
}

void PoolingGenConn::readSlownessFlag(PVParams * params) {
   slownessFlag = params->value(name, "slownessFlag", 0.0/*default is false*/, true/*warn if absent*/);
}

int PoolingGenConn::readSlownessPre(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "slownessFlag"));
   int status = slownessFlag ? getSlownessLayer(&slownessPre, "slownessPre") : PV_SUCCESS;
   return status;
}

int PoolingGenConn::readSlownessPost(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "slownessFlag"));
   int status = slownessFlag ? getSlownessLayer(&slownessPost, "slownessPost") : PV_SUCCESS;
   return status;
}

int PoolingGenConn::communicateInitInfo() {
   int status = GenerativeConn::communicateInitInfo();
   if (status != PV_SUCCESS) return status;
   pre2 = parent->getLayerFromName(preLayerName2);
   post2 = parent->getLayerFromName(postLayerName2);
   if (status == PV_SUCCESS) {
      status = checkLayersCompatible(pre, pre2) && checkLayersCompatible(post, post2) ? PV_SUCCESS : PV_FAILURE;
   }
   else {
      status = PV_FAILURE;
   }
   if( status == PV_SUCCESS ) {
      slownessFlag = parent->parameters()->value(name, "slownessFlag", 0.0/*default is false*/);
   }
   if( slownessFlag ) {
      status = getSlownessLayer(&slownessPre, "slownessPre");
      status = getSlownessLayer(&slownessPost, "slownessPost")==PV_SUCCESS ? status : PV_FAILURE;
   }
   if( slownessFlag && status == PV_SUCCESS ) {
      status = checkLayersCompatible(pre, slownessPre) ? status : PV_FAILURE;
      status = checkLayersCompatible(post, slownessPost) ? status : PV_FAILURE;
   }
   if( status != PV_SUCCESS ) {
      abort();
   }
   return status;
}

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

int PoolingGenConn::getSlownessLayer(HyPerLayer ** l, const char * paramname) {
   int status = PV_SUCCESS;
   assert(slownessFlag);
   const char * slownessLayerName = parent->parameters()->stringValue(name, paramname, false);
   if( slownessLayerName == NULL ) {
      status = PV_FAILURE;
      fprintf(stderr, "PoolingGenConn \"%s\": if slownessFlag is set, parameter \"%s\" must be set\n", name, paramname);
   }
   if( status == PV_SUCCESS ) {
      *l = parent->getLayerFromName(slownessLayerName);
      if( *l == NULL ) {
         status = PV_FAILURE;
         fprintf(stderr, "PoolingGenConn \"%s\": %s layer \"%s\" was not found\n", name, paramname, slownessLayerName);
      }
   }
   return status;
}

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
        int nk = weights->nx * nfp;
        pvdata_t * postactRef = &(postSynapticLayer()->getCLayer()->activity->data[offset]);
        pvdata_t * postact2Ref = &(getPost2()->getCLayer()->activity->data[offset]);
        int sya = getPostNonextStrides()->sy;
        pvdata_t * wtpatch = get_wData(axonID, kExt); // weights->data;
        int syw = syp;
        for( int y=0; y<nyp; y++ ) {
            int lineoffsetw = 0;
            int lineoffseta = 0;
            for( int k=0; k<nk; k++ ) {
                float w = wtpatch[lineoffsetw + k] + relaxation*(preact*postactRef[lineoffseta + k]+preact2*postact2Ref[lineoffseta + k]);
                wtpatch[lineoffsetw + k] = w;
            }
            lineoffsetw += syw;
            lineoffseta += sya;
        }
    }
    if( slownessFlag ) {
       for(int kPre=0; kPre<nPre;kPre++) {
           int kExt = kIndexExtended(kPre, nx, ny, nf, pad);

           size_t offset = getAPostOffset(kPre, axonID);
           pvdata_t preact = slownessPre->getCLayer()->activity->data[kExt];
           PVPatch * weights = getWeights(kPre,axonID);
           int nyp = weights->ny;
           int nk = weights->nx * nfp;
           pvdata_t * postactRef = &(slownessPost->getCLayer()->activity->data[offset]);
           int sya = getPostNonextStrides()->sy;
           pvdata_t * wtpatch = get_wData(axonID, kExt); // weights->data;
           int syw = syp;
           for( int y=0; y<nyp; y++ ) {
               int lineoffsetw = 0;
               int lineoffseta = 0;
               for( int k=0; k<nk; k++ ) {
                   float w = wtpatch[lineoffsetw + k] - relaxation*(preact*postactRef[lineoffseta + k]);
                   wtpatch[lineoffsetw + k] = w;
               }
               lineoffsetw += syw;
               lineoffseta += sya;
           }
       }
    }
    if( nonnegConstraintFlag ) {
       for(int kPatch=0; kPatch<getNumDataPatches();kPatch++) {
          // PVPatch * weights = this->getKernelPatch(axonID, kPatch);
          pvdata_t * wtpatch = get_wDataHead(axonID, kPatch); // weights->data;
           int nk = nxp * nfp;
           int syw = nxp*nfp;
           for( int y=0; y < nyp; y++ ) {
               int lineoffsetw = 0;
               for( int k=0; k<nk; k++ ) {
                   pvdata_t w = wtpatch[lineoffsetw + k];
                   if( w<0 ) {
                      wtpatch[lineoffsetw + k] = 0;
                   }
               }
               lineoffsetw += syw;
           }
       }
    }
    // normalizeWeights now called in KernelConn::updateState
    lastUpdateTime = parent->simulationTime();

    return PV_SUCCESS;
}

PoolingGenConn::~PoolingGenConn() {
   free(preLayerName2); preLayerName2 = NULL;
   free(postLayerName2); postLayerName2 = NULL;
}

}  // end namespace PV
