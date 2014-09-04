/*
 * PoolingGenConn.cpp
 *
 *  Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#include "PoolingGenConn.hpp"

namespace PV {
PoolingGenConn::PoolingGenConn(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}  // end of PoolingGenConn::PoolingGenConn(const char *, HyPerCol *)

int PoolingGenConn::initialize_base() {
    pre2 = NULL;
    post2 = NULL;
    return PV_SUCCESS;
}

int PoolingGenConn::initialize(const char * name, HyPerCol * hc) {
   int status = GenerativeConn::initialize(name, hc);
   return status;
}

int PoolingGenConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = GenerativeConn::ioParamsFillGroup(ioFlag);
   ioParam_secondaryPreLayerName(ioFlag);
   ioParam_secondaryPostLayerName(ioFlag);
   ioParam_slownessFlag(ioFlag);
   ioParam_slownessPre(ioFlag);
   ioParam_slownessPost(ioFlag);
   return status;
}

void PoolingGenConn::ioParam_secondaryPreLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "secondaryPreLayerName", &preLayerName2);
}

void PoolingGenConn::ioParam_secondaryPostLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "secondaryPostLayerName", &postLayerName2);
}

void PoolingGenConn::ioParam_slownessFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "slownessFlag", &slownessFlag, false/*default value*/, true/*warnIfAbsent*/);
}

void PoolingGenConn::ioParam_slownessPre(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "slownessFlag"));
   if (slownessFlag) parent->ioParamStringRequired(ioFlag, name, "slownessPre", &slownessPreName);
}

void PoolingGenConn::ioParam_slownessPost(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "slownessFlag"));
   if (slownessFlag) parent->ioParamStringRequired(ioFlag, name, "slownessPost", &slownessPostName);
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
   assert(!parent->parameters()->presentAndNotBeenRead(name, "slownessFlag"));
   if( slownessFlag ) {
      slownessPre = parent->getLayerFromName(slownessPreName);
      slownessPost = parent->getLayerFromName(slownessPostName);
      if (slownessPre==NULL || slownessPost==NULL) {
         status = PV_FAILURE;
         if (slownessPre==NULL && parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: slownessPre layer \"%s\" is not a layer in the column.\n",
                  parent->parameters()->groupKeywordFromName(name), name, slownessPreName);
         }
         if (slownessPost==NULL && parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: slownessPost layer \"%s\" is not a layer in the column.\n",
                  parent->parameters()->groupKeywordFromName(name), name, slownessPostName);
         }
      }
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
}  // end of PoolingGenConn::PoolingGenConn(HyPerLayer *, HyPerLayer *)

int PoolingGenConn::getSlownessLayerName(char ** l, const char * paramname) {
   int status = PV_SUCCESS;
   assert(slownessFlag);
   const char * slownessLayerName = parent->parameters()->stringValue(name, paramname, false);
   if( slownessLayerName == NULL ) {
      status = PV_FAILURE;
      fprintf(stderr, "PoolingGenConn \"%s\": if slownessFlag is set, parameter \"%s\" must be set\n", name, paramname);
   }
   if( status == PV_SUCCESS ) {
      *l = strdup(slownessLayerName);
      if( *l == NULL ) {
         status = PV_FAILURE;
         fprintf(stderr, "%s \"%s\": error allocating memory for %s: %s\n",
               parent->parameters()->groupKeywordFromName(name), name, paramname, strerror(errno));
      }
   }
   return status;
}

int PoolingGenConn::updateWeights(int axonID) {
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
        pvwdata_t * wtpatch = get_wData(axonID, kExt); // weights->data;
        int syw = syp;
        for( int y=0; y<nyp; y++ ) {
            int lineoffsetw = 0;
            int lineoffseta = 0;
            for( int k=0; k<nk; k++ ) {
                pvdata_t w = wtpatch[lineoffsetw + k] + relaxation*(preact*postactRef[lineoffseta + k]+preact2*postact2Ref[lineoffseta + k]);
                wtpatch[lineoffsetw + k] = w;
            }
            lineoffsetw += syw;
            lineoffseta += sya;
        }
    }
    if( slownessFlag ) {
       for(int kPre=0; kPre<nPre;kPre++) {
           int kExt = kIndexExtended(kPre, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);

           size_t offset = getAPostOffset(kPre, axonID);
           pvdata_t preact = slownessPre->getCLayer()->activity->data[kExt];
           PVPatch * weights = getWeights(kPre,axonID);
           int nyp = weights->ny;
           int nk = weights->nx * nfp;
           pvdata_t * postactRef = &(slownessPost->getCLayer()->activity->data[offset]);
           int sya = getPostNonextStrides()->sy;
           pvwdata_t * wtpatch = get_wData(axonID, kExt); // weights->data;
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
           pvwdata_t * wtpatch = get_wDataHead(axonID, kPatch); // weights->data;
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
    // normalizeWeights now called in HyPerConn::updateState
    lastUpdateTime = parent->simulationTime();

    return PV_SUCCESS;
}

PoolingGenConn::~PoolingGenConn() {
   free(preLayerName2); preLayerName2 = NULL;
   free(postLayerName2); postLayerName2 = NULL;
}

}  // end namespace PV
