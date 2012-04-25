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

int InitTransposeWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex,
                           int arborID, InitWeightsParams *weightParams) {

   InitTransposeWeightsParams *weightParamPtr = dynamic_cast<InitTransposeWeightsParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }
   if(arborID!=0) {
      fprintf(stderr, "InitTransposeWeights not yet implemented for connections with multiple arbors.\n");
      exit(EXIT_FAILURE);
   }


   weightParamPtr->calcOtherParams(patchIndex);

   transposeKernels(dataStart, patchIndex, weightParamPtr);

   return PV_SUCCESS; // return 1;

}

int InitTransposeWeights::transposeKernels(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, InitTransposeWeightsParams * weightParamPtr) {
   // compute the transpose of originalConn->kernelPatches and
   // store into this->kernelPatches
   // assume scale factors are 1 and that nxp, nyp are odd.


   HyPerLayer *pre = weightParamPtr->getPre();
   HyPerLayer *post = weightParamPtr->getPost();
   KernelConn *thisConn = static_cast<KernelConn*> (weightParamPtr->getParentConn());
   assert(thisConn!=NULL);
   assert(originalConn->numberOfAxonalArborLists()==1); // If more than one arbor, the arborId would need to be passed as an input.

   int nyPatch_tmp = weightParamPtr->getnyPatch_tmp();
   int nxPatch_tmp = weightParamPtr->getnxPatch_tmp();

   int xscalediff = pre->getXScale()-post->getXScale();
   int yscalediff = pre->getYScale()-post->getYScale();
   // scalediff>0 means TransposeConn's post--that is, the originalConn's pre--has a higher neuron density

   int numFBKernelPatches = thisConn->getNumDataPatches();
   int numFFKernelPatches = originalConn->getNumDataPatches();

   if( xscalediff <= 0 && yscalediff <= 0) {
      int xscaleq = (int) powf(2,-xscalediff);
      int yscaleq = (int) powf(2,-yscalediff);
      assert(numFBKernelPatches == originalConn->fPatchSize() * xscaleq * yscaleq);

//   for( int kernelnumberFB = 0; kernelnumberFB < numFBKernelPatches; kernelnumberFB++ ) {
      // PVPatch * kpFB = thisConn->getKernelPatch(0, kernelnumberFB);
      int nfFB = thisConn->fPatchSize();
      assert(numFFKernelPatches == nfFB);
      int nxFB = thisConn->xPatchSize(); // kpFB->nx;
      int nyFB = thisConn->yPatchSize(); // kpFB->ny;
      for( int kyFB = 0; kyFB < nyFB; kyFB++ ) {
         for( int kxFB = 0; kxFB < nxFB; kxFB++ ) {
            for( int kfFB = 0; kfFB < nfFB; kfFB++ ) {
               int kIndexFB = kIndex(kxFB,kyFB,kfFB,nxFB,nyFB,nfFB);
               int kernelnumberFF = kfFB;
               int patchSizeFF = originalConn->xPatchSize()*originalConn->yPatchSize()*originalConn->fPatchSize();
               pvdata_t * dataStartFF = thisConn->get_wDataStart(0)+kernelnumberFF*patchSizeFF; // PVPatch * kpFF = originalConn->getKernelPatch(0, kernelnumberFF);
               int kfFF = featureIndex(patchIndex, xscaleq, yscaleq, originalConn->fPatchSize());
               int kxFFoffset = kxPos(patchIndex, xscaleq, yscaleq, originalConn->fPatchSize());
               int kxFF = (nxPatch_tmp - 1 - kxFB) * xscaleq + kxFFoffset;
               int kyFFoffset = kyPos(patchIndex, xscaleq, yscaleq, originalConn->fPatchSize());
               int kyFF = (nyPatch_tmp - 1 - kyFB) * yscaleq + kyFFoffset;
               int kIndexFF = kIndex(kxFF, kyFF, kfFF, originalConn->xPatchSize(), originalConn->yPatchSize(), originalConn->fPatchSize());
               // can the calls to kxPos, kyPos, featureIndex be replaced by one call to patchIndexToKernelIndex?
               dataStart[kIndexFB] = dataStartFF[kIndexFF];
            }
         }
      }
//   }
   }
   else if( xscalediff > 0 && yscalediff > 0) {
      int xscaleq = (int) powf(2,xscalediff);
      int yscaleq = (int) powf(2,yscalediff);
//   for( int kernelnumberFB = 0; kernelnumberFB < numFBKernelPatches; kernelnumberFB++ ) {
      // PVPatch * kpFB = thisConn->getKernelPatch(0, kernelnumberFB);
      int nxFB = thisConn->xPatchSize(); // kpFB->nx;
      int nyFB = thisConn->yPatchSize(); // kpFB->ny;
      int nfFB = thisConn->fPatchSize();
      for( int kyFB = 0; kyFB < nyFB; kyFB++ ) {
         int precelloffsety = kyFB % yscaleq;
         for( int kxFB = 0; kxFB < nxFB; kxFB++ ) {
            int precelloffsetx = kxFB % xscaleq;
            for( int kfFB = 0; kfFB < nfFB; kfFB++ ) {
               int kernelnumberFF = (precelloffsety*xscaleq + precelloffsetx)*nfFB + kfFB;
               int patchSizeFF = originalConn->xPatchSize()*originalConn->yPatchSize()*originalConn->fPatchSize();
               pvdata_t * dataStartFF = thisConn->get_wDataStart(0)+kernelnumberFF*patchSizeFF; // PVPatch * kpFF = originalConn->getKernelPatch(0, kernelnumberFF);
               int kxFF = (nxPatch_tmp-kxFB-1)/xscaleq;
               assert(kxFF >= 0 && kxFF < originalConn->xPatchSize());
               int kyFF = (nyPatch_tmp-kyFB-1)/yscaleq;
               assert(kyFF >= 0 && kyFF < originalConn->yPatchSize());
               int kfFF = patchIndex;
               assert(kfFF >= 0 && kfFF < originalConn->fPatchSize());
               int kIndexFF = kIndex(kxFF, kyFF, kfFF, originalConn->xPatchSize(), originalConn->yPatchSize(), originalConn->fPatchSize());
               int kIndexFB = kIndex(kxFB, kyFB, kfFB, nxFB, nyFB, nfFB);
               dataStart[kIndexFB] = dataStartFF[kIndexFF];
            }
         }
      }
//   }
   }
   else {
       fprintf(stderr,"xscalediff = %d, yscalediff = %d: the case of many-to-one in one dimension and one-to-many in the other"
                      "has not yet been implemented.\n", xscalediff, yscalediff);
       exit(1);
   }

   return PV_SUCCESS;}

} /* namespace PV */
