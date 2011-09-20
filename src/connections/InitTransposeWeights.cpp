/*
 * InitTransposeWeights.cpp
 *
 *  Created on: Aug 15, 2011
 *      Author: kpeterson
 */

#include "InitTransposeWeights.hpp"
#include "InitTransposeWeightsParams.hpp"

namespace PV {

InitTransposeWeights::InitTransposeWeights()
{
   initialize_base();
}

InitTransposeWeights::InitTransposeWeights(KernelConn * origConn)
{
   initialize_base();
   initialize(origConn);
}

InitTransposeWeights::~InitTransposeWeights()
{
   // TODO Auto-generated destructor stub
}

int InitTransposeWeights::initialize_base() {
   return PV_SUCCESS;
}
int InitTransposeWeights::initialize(KernelConn * origConn) {
   assert(origConn!=NULL);
   originalConn = origConn;
   return PV_SUCCESS;
}

InitWeightsParams * InitTransposeWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitTransposeWeightsParams(callingConn);
   return tempPtr;
}

int InitTransposeWeights::calcWeights(PVPatch * patch, int patchIndex,
                           InitWeightsParams *weightParams) {

   InitTransposeWeightsParams *weightParamPtr = dynamic_cast<InitTransposeWeightsParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }


   weightParamPtr->calcOtherParams(patch, patchIndex);

   transposeKernels(patch, weightParamPtr);

   return PV_SUCCESS; // return 1;

}

int InitTransposeWeights::transposeKernels(PVPatch * patch, InitTransposeWeightsParams * weightParamPtr) {
   // compute the transpose of originalConn->kernelPatches and
   // store into this->kernelPatches
   // assume scale factors are 1 and that nxp, nyp are odd.

   HyPerLayer *pre = weightParamPtr->getPre();
   HyPerLayer *post = weightParamPtr->getPost();
   KernelConn *thisConn = static_cast<KernelConn*> (weightParamPtr->getParentConn());
   assert(thisConn!=NULL);

   int nyPatch_tmp = weightParamPtr->getnyPatch_tmp();
   int nxPatch_tmp = weightParamPtr->getnxPatch_tmp();

   int xscalediff = pre->getXScale()-post->getXScale();
   int yscalediff = pre->getYScale()-post->getYScale();
   // scalediff>0 means TransposeConn's post--that is, the originalConn's pre--has a higher neuron density

   int numFBKernelPatches = thisConn->numDataPatches();
   int numFFKernelPatches = originalConn->numDataPatches();

   if( xscalediff <= 0 && yscalediff <= 0) {
       int xscaleq = (int) powf(2,-xscalediff);
       int yscaleq = (int) powf(2,-yscalediff);

       for( int kernelnumberFB = 0; kernelnumberFB < numFBKernelPatches; kernelnumberFB++ ) {
           PVPatch * kpFB = thisConn->getKernelPatch(0, kernelnumberFB);
           int nfFB = kpFB->nf;
              assert(numFFKernelPatches == nfFB);
           int nxFB = kpFB->nx;
           int nyFB = kpFB->ny;
           for( int kyFB = 0; kyFB < nyFB; kyFB++ ) {
               for( int kxFB = 0; kxFB < nxFB; kxFB++ ) {
                   for( int kfFB = 0; kfFB < nfFB; kfFB++ ) {
                       int kIndexFB = kIndex(kxFB,kyFB,kfFB,nxFB,nyFB,nfFB);
                       int kernelnumberFF = kfFB;
                       PVPatch * kpFF = originalConn->getKernelPatch(0, kernelnumberFF);
                          assert(numFBKernelPatches == kpFF->nf * xscaleq * yscaleq);
                       int kfFF = featureIndex(kernelnumberFB, xscaleq, yscaleq, originalConn->fPatchSize());
                       int kxFFoffset = kxPos(kernelnumberFB, xscaleq, yscaleq, originalConn->fPatchSize());
                       int kxFF = (nxPatch_tmp - 1 - kxFB) * xscaleq + kxFFoffset;
                       int kyFFoffset = kyPos(kernelnumberFB, xscaleq, yscaleq, originalConn->fPatchSize());
                       int kyFF = (nyPatch_tmp - 1 - kyFB) * yscaleq + kyFFoffset;
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
           PVPatch * kpFB = thisConn->getKernelPatch(0, kernelnumberFB);
           int nxFB = kpFB->nx;
           int nyFB = kpFB->ny;
           int nfFB = kpFB->nf;
           for( int kyFB = 0; kyFB < nyFB; kyFB++ ) {
               int precelloffsety = kyFB % yscaleq;
                  for( int kxFB = 0; kxFB < nxFB; kxFB++ ) {
                   int precelloffsetx = kxFB % xscaleq;
                   for( int kfFB = 0; kfFB < nfFB; kfFB++ ) {
                       int kernelnumberFF = (precelloffsety*xscaleq + precelloffsetx)*nfFB + kfFB;
                       PVPatch * kpFF = originalConn->getKernelPatch(0, kernelnumberFF);
                       int kxFF = (nxPatch_tmp-kxFB-1)/xscaleq;
                       assert(kxFF >= 0 && kxFF < originalConn->xPatchSize());
                       int kyFF = (nyPatch_tmp-kyFB-1)/yscaleq;
                       assert(kyFF >= 0 && kyFF < originalConn->yPatchSize());
                       int kfFF = kernelnumberFB;
                       assert(kfFF >= 0 && kfFF < originalConn->fPatchSize());
                       int kIndexFF = kIndex(kxFF, kyFF, kfFF, kpFF->nx, kpFF->ny, kpFF->nf);
                       int kIndexFB = kIndex(kxFB, kyFB, kfFB, nxFB, nyFB, nfFB);
                       kpFB->data[kIndexFB] = kpFF->data[kIndexFF];
                   }
               }
           }
       }
   }
   else {
       fprintf(stderr,"xscalediff = %d, yscalediff = %d: the case of many-to-one in one dimension and one-to-many in the other"
                      "has not yet been implemented.\n", xscalediff, yscalediff);
       exit(1);
   }

   return PV_SUCCESS;}

} /* namespace PV */
