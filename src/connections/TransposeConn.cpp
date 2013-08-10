/*
 * TransposeConn.cpp
 *
 *  Created on: May 16, 2011
 *      Author: peteschultz
 */

#include "TransposeConn.hpp"

namespace PV {

TransposeConn::TransposeConn() {
   initialize_base();
}  // TransposeConn::~TransposeConn()

TransposeConn::TransposeConn(const char * name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name, const char * originalConnName) {
   initialize_base();
   int status = initialize(name, hc, pre_layer_name, post_layer_name, originalConnName);
   assert(status == PV_SUCCESS);
}  // TransposeConn::TransposeConn(const char * name, HyPerCol * hc, HyPerLayer *, HyPerLayer *, ChannelType, KernelConn *)

TransposeConn::~TransposeConn() {

}  // TransposeConn::~TransposeConn()

int TransposeConn::initialize_base() {
   plasticityFlag = true; // Default value; override in params
   weightUpdatePeriod = 1;   // Default value; override in params
   // TransposeConn::initialize_base() gets called after
   // KernelConn::initialize_base() so these default values override
   // those in KernelConn::initialize_base().
   // TransposeConn::initialize_base() gets called before
   // KernelConn::initialize(), so these values still get overridden
   // by the params file values.

   originalConnName = NULL;
   originalConn = NULL;
   return PV_SUCCESS;
}  // TransposeConn::initialize_base()

int TransposeConn::initialize(const char * name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name, const char * originalConnName) {
   int status = PV_SUCCESS;
   status = KernelConn::initialize(name, hc, pre_layer_name, post_layer_name, NULL, NULL);
   if (originalConnName == NULL) {
      fprintf(stderr, "TransposeConn \"%s\" error in rank %d: originalConnName must be set.\n", name, parent->columnId());
      status = PV_FAILURE;
      return status;
   }
   this->originalConnName = strdup(originalConnName);
   if (originalConnName == NULL) {
      fprintf(stderr, "TransposeConn \"%s\" error in rank %d: unable to allocate memory for originalConnName \"%s\": %s\n",
            name, parent->columnId(), originalConnName, strerror(errno));
   }
   return status;
}

void TransposeConn::readNumAxonalArbors(PVParams * params) {
   // numAxonalArbors will be copied from originalConn.
   // However, readNumAxonalArbors is called during initialize,
   // and originalConn is set during communicate, so we have to wait.
   // Need to override so that numAxonalArbors doesn't get read from params.
}

int TransposeConn::readPatchSize(PVParams * params) {
   // During the communication phase, nxp, nyp, nxpShrunken, nypShrunken will be determined from originalConn
   return PV_SUCCESS;
}
/*
int TransposeConn::readPatchSize(PVParams * params) {
   // If originalConn is many-to-one, the transpose connection is one-to-many; then xscaleDiff > 0.
   // Similarly, if originalConn is one-to-many, xscaleDiff < 0.

   // Since this is called before pre, post and originalConn are set up, we have to load nxp,nyp
   // and compute the scale differences instead of grabbing them from the object.
   // The problem with waiting until after pre, post and originalConn are defined (in communicateInitInfo())
   // is that some of the things in communicateInitInfo()  (e.g. requireMarginWidth) use nxp and nyp
   // before communicate gets called.

   // Some of the code duplication might be eliminated by adding some functions to convert.h

   char * origPreName = NULL;
   char * origPostName = NULL;
   HyPerConn::getPreAndPostLayerNames(originalConnName, params, &origPreName, &origPostName);
   if(origPreName==NULL || origPostName==NULL) {
      exit(EXIT_FAILURE); // getPreAndPostLayerNames printed error messages
   }

   float origXScalePre = params->value(origPreName, "nxScale", 1.0f);
   float origXScalePost = params->value(origPostName, "nxScale", 1.0f);
   int xscaleDiff = (int) nearbyint(-log2( (double) (origXScalePost))) - (int) nearbyint(-log2( (double) (origXScalePre))); // post-pre because the feedback connection goes the other way.
   int nxp_orig = params->value(originalConnName, "nxp", 1);
   nxp = nxp_orig;
   if(xscaleDiff > 0 ) {
      nxp *= (int) pow( 2, xscaleDiff );
   }
   else if(xscaleDiff < 0) {
      nxp /= (int) pow(2,-xscaleDiff);
      assert(nxp_orig==nxp*pow( 2, (float) (-xscaleDiff) ));
   }

   float origYScalePre = params->value(origPreName, "nyScale", 1.0f);
   float origYScalePost = params->value(origPostName, "nyScale", 1.0f);
   int yscaleDiff = (int) nearbyint(-log2( (double) (origYScalePost))) - (int) nearbyint(-log2( (double) (origYScalePre)));
   int nyp_orig = params->value(originalConnName, "nyp", 1);
   nyp = nyp_orig;
   if(yscaleDiff > 0 ) {
      nyp *= (int) pow( 2, (float) yscaleDiff );
   }
   else if(yscaleDiff < 0) {
      nyp /= (int) pow(2,-yscaleDiff);
      assert(nyp_orig==nyp*pow( 2, (float) (-yscaleDiff) ));
   }

   free(origPreName);
   free(origPostName);

   nxpShrunken = nxp;
   nypShrunken = nyp;
   return PV_SUCCESS;
}
*/

int TransposeConn::readNfp(PVParams * params) {
   // Empty override since nfp will be inferred from originalConn
   return PV_SUCCESS;
}

void TransposeConn::readPlasticityFlag(PVParams * params) {
   // Empty override since plasticityFlag will be copied from originalConn
}

void TransposeConn::readCombine_dW_with_W_flag(PVParams * params) {
   combine_dW_with_W_flag = false;
}

void TransposeConn::read_dWMax(PVParams * params) {
   dWMax = 1.0;
}

void TransposeConn::readKeepKernelsSynchronized(PVParams * params) {
   keepKernelsSynchronized_flag = false;
}

void TransposeConn::readWeightUpdatePeriod(PVParams * params) {
   weightUpdatePeriod = 0.0f;  // Ensures that every timestep updateState calls updateWeights, which will compare lastUpdateTime to originalConn's lastUpdateTime
}

void TransposeConn::readInitialWeightUpdateTime(PVParams * params) {
   weightUpdateTime = parent->simulationTime();
}

void TransposeConn::readShrinkPatches(PVParams * params) {
   // Will check that originalConn has shrinkPatches set to false in communicate phase, once originalConn is set.
   // Need to override here since overridden method reads shrinkPatches from params.
   shrinkPatches_flag = false;
}

InitWeights * TransposeConn::handleMissingInitWeights(PVParams * params) {
   // TransposeConn doesn't use InitWeights; it initializes the weight by transposing the initial weights of originalConn
   return NULL;
}

int TransposeConn::communicateInitInfo() {
   int status = PV_SUCCESS;
   HyPerConn * c = parent->getConnFromName(originalConnName);
   if (c == NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "TransposeConn \"%s\" error: originalConnName \"%s\" is not an existing connection.\n", name, originalConnName);
         status = PV_FAILURE;
      }
   }
   if (status != PV_SUCCESS) return status;
   originalConn = dynamic_cast<KernelConn *>(c);
   if (originalConn==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "TransposeConn \"%s\" error: originalConnName \"%s\" must be a KernelConn or a KernelConn-derived class.\n", name, originalConnName);
         status = PV_FAILURE;
      }
   }
   if (status != PV_SUCCESS) return status;

   status = KernelConn::communicateInitInfo();
   if (status != PV_SUCCESS) return status;

   numAxonalArborLists = originalConn->numberOfAxonalArborLists();
   //TransposeConn has not been updated to support multiple arbors
   //if (numAxonalArborLists!=1) {
   //   if (parent->columnId()==0) {
   //      fprintf(stderr, "TransposeConn error for connection \"%s\": Currently, originalConn \"%s\" can have only one arbor.\n", name, originalConn->getName());
   //   }
   //   MPI_Barrier(getParent()->icCommunicator()->communicator());
   //   exit(EXIT_FAILURE);
   //}
   plasticityFlag = originalConn->getPlasticityFlag();

   if(originalConn->getShrinkPatches_flag()) {
      if (parent->columnId()==0) {
         fprintf(stderr, "TransposeConn \"%s\" error: original conn \"%s\" has shrinkPatches set to true.  TransposeConn has not been implemented for that case.\n", name, originalConn->getName());
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   return status;
}

int TransposeConn::setPatchSize() {
   // If originalConn is many-to-one, the transpose connection is one-to-many; then xscaleDiff > 0.
   // Similarly, if originalConn is one-to-many, xscaleDiff < 0.

   // Since this is called before pre, post and originalConn are set up, we have to load nxp,nyp
   // and compute the scale differences instead of grabbing them from the object.
   // The problem with waiting until after pre, post and originalConn are defined (in communicateInitInfo())
   // is that some of the things in communicateInitInfo()  (e.g. requireMarginWidth) use nxp and nyp
   // before communicate gets called.

   // Some of the code duplication might be eliminated by adding some functions to convert.h

   assert(pre && post);
   assert(originalConn);

   int xscaleDiff = pre->getXScale() - post->getXScale();
   int nxp_orig = originalConn->xPatchSize();
   nxp = nxp_orig;
   if(xscaleDiff > 0 ) {
      nxp *= (int) pow( 2, xscaleDiff );
   }
   else if(xscaleDiff < 0) {
      nxp /= (int) pow(2,-xscaleDiff);
      assert(nxp_orig==nxp*pow( 2, (float) (-xscaleDiff) ));
   }

   int yscaleDiff = pre->getYScale() - post->getYScale();
   int nyp_orig = originalConn->yPatchSize();
   nyp = nyp_orig;
   if(yscaleDiff > 0 ) {
      nyp *= (int) pow( 2, yscaleDiff );
   }
   else if(yscaleDiff < 0) {
      nyp /= (int) pow(2,-yscaleDiff);
      assert(nyp_orig==nyp*pow( 2, (float) (-yscaleDiff) ));
   }

   nfp = post->getLayerLoc()->nf;
   assert(nfp==originalConn->preSynapticLayer()->getLayerLoc()->nf);

   nxpShrunken = nxp;
   nypShrunken = nyp;
   return PV_SUCCESS;

}

int TransposeConn::allocateDataStructures() {
   KernelConn::allocateDataStructures();
   // Don't need to call KernelConn::allocateDataStructures(), since all it does is create mpiReductionBuffer, but transpose is automatically reduced if the originalConn is reduced.
   normalizer = NULL;
   // normalize_flag = false; // replaced by testing whether normalizer!=NULL
   return PV_SUCCESS;
}

PVPatch *** TransposeConn::initializeWeights(PVPatch *** arbors, pvdata_t ** dataStart, int numPatches, const char * filename) {
   if( filename ) {
      return KernelConn::initializeWeights(arbors, dataStart, numPatches, filename);
   }
   else {
      for(int arborId = 0; arborId < numAxonalArborLists; arborId++){
         transposeKernels(arborId);
      }
   }
   return arbors;
}  // TransposeConn::initializeWeights(PVPatch **, int, const char *)

// int TransposeConn::initNormalize() {
//    normalize_flag = false;
//    return PV_SUCCESS;
// }

int TransposeConn::updateWeights(int axonID) {
   int status;
   float original_update_time = originalConn->getLastUpdateTime();
   if(original_update_time > lastUpdateTime ) {
      status = transposeKernels(axonID);
      lastUpdateTime = parent->simulationTime();
   }
   else
      status = PV_SUCCESS;
   return status;
}  // end of TransposeConn::updateWeights(int);

// TODO reorganize transposeKernels():  Loop over kernelNumberFB on outside, call transposeOneKernel(kpFB, kernelnumberFB), which handles all the cases.
// This would play better with Kris's initWeightsMethod.
int TransposeConn::transposeKernels(int arborId) {
   // compute the transpose of originalConn->kernelPatches and
   // store into this->kernelPatches
   // assume scale factors are 1 and that nxp, nyp are odd.

   int xscalediff = pre->getXScale()-post->getXScale();
   int yscalediff = pre->getYScale()-post->getYScale();
   // scalediff>0 means TransposeConn's post--that is, the originalConn's pre--has a higher neuron density

   int numFBKernelPatches = getNumDataPatches();
   int numFFKernelPatches = originalConn->getNumDataPatches();

   if( xscalediff <= 0 && yscalediff <= 0) {
      int xscaleq = (int) pow(2,-xscalediff);
      int yscaleq = (int) pow(2,-yscalediff);


      for( int kernelnumberFB = 0; kernelnumberFB < numFBKernelPatches; kernelnumberFB++ ) {
         // PVPatch * kpFB = getKernelPatch(0, kernelnumberFB);
         pvdata_t * dataStartFB = get_wDataHead(arborId, kernelnumberFB);
         int nfFB = nfp;
         assert(numFFKernelPatches == nfFB);
         int nxFB = nxp; // kpFB->nx;
         int nyFB = nyp; // kpFB->ny;
         for( int kyFB = 0; kyFB < nyFB; kyFB++ ) {
            for( int kxFB = 0; kxFB < nxFB; kxFB++ ) {
               for( int kfFB = 0; kfFB < nfFB; kfFB++ ) {
                  int kIndexFB = kIndex(kxFB,kyFB,kfFB,nxFB,nyFB,nfFB);
                  int kernelnumberFF = kfFB;
                  // PVPatch * kpFF = originalConn->getKernelPatch(0, kernelnumberFF);
                  pvdata_t * dataStartFF = originalConn->get_wDataHead(arborId, kernelnumberFF);
                  int nxpFF = originalConn->xPatchSize();
                  int nypFF = originalConn->yPatchSize();
                  assert(numFBKernelPatches == originalConn->fPatchSize() * xscaleq * yscaleq);
                  int kfFF = featureIndex(kernelnumberFB, xscaleq, yscaleq, originalConn->fPatchSize());
                  int kxFFoffset = kxPos(kernelnumberFB, xscaleq, yscaleq, originalConn->fPatchSize());
                  int kxFF = (nxp - 1 - kxFB) * xscaleq + kxFFoffset;
                  int kyFFoffset = kyPos(kernelnumberFB, xscaleq, yscaleq, originalConn->fPatchSize());
                  int kyFF = (nyp - 1 - kyFB) * yscaleq + kyFFoffset;
                  int kIndexFF = kIndex(kxFF, kyFF, kfFF, nxpFF, nypFF, originalConn->fPatchSize());
                  // can the calls to kxPos, kyPos, featureIndex be replaced by one call to patchIndexToKernelIndex?
                  dataStartFB[kIndexFB] = dataStartFF[kIndexFF];
                  // kpFB->data[kIndexFB] = kpFF->data[kIndexFF];
               }
            }
         }
      }
   }
   else if( xscalediff > 0 && yscalediff > 0) {
      int xscaleq = (int) pow(2,xscalediff);
      int yscaleq = (int) pow(2,yscalediff);
      for( int kernelnumberFB = 0; kernelnumberFB < numFBKernelPatches; kernelnumberFB++ ) {
         // PVPatch * kpFB = getKernelPatch(0, kernelnumberFB);
         pvdata_t * dataStartFB = get_wDataHead(arborId, kernelnumberFB);
         int nxFB = nxp; // kpFB->nx;
         int nyFB = nyp; // kpFB->ny;
         int nfFB = nfp;
         for( int kyFB = 0; kyFB < nyFB; kyFB++ ) {
            int precelloffsety = kyFB % yscaleq;
            for( int kxFB = 0; kxFB < nxFB; kxFB++ ) {
               int precelloffsetx = kxFB % xscaleq;
               for( int kfFB = 0; kfFB < nfFB; kfFB++ ) {
                  int kernelnumberFF = (precelloffsety*xscaleq + precelloffsetx)*nfFB + kfFB;
                  pvdata_t * dataStartFF = originalConn->get_wDataHead(arborId, kernelnumberFF);
                  int nxpFF = originalConn->xPatchSize();
                  int nypFF = originalConn->yPatchSize();
                  int kxFF = (nxp-kxFB-1)/xscaleq;
                  assert(kxFF >= 0 && kxFF < originalConn->xPatchSize());
                  int kyFF = (nyp-kyFB-1)/yscaleq;
                  assert(kyFF >= 0 && kyFF < originalConn->yPatchSize());
                  int kfFF = kernelnumberFB;
                  assert(kfFF >= 0 && kfFF < originalConn->fPatchSize());
                  int kIndexFF = kIndex(kxFF, kyFF, kfFF, nxpFF, nypFF, originalConn->fPatchSize());
                  int kIndexFB = kIndex(kxFB, kyFB, kfFB, nxFB, nyFB, nfFB);
                  dataStartFB[kIndexFB] = dataStartFF[kIndexFF];
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

   return PV_SUCCESS;
}  // TransposeConn::transposeKernels()

int TransposeConn::reduceKernels(int arborID) {
   // Values are taken from originalConn.  If originalConn keeps kernels synchronized, then TransposeConn stays synchronized automatically.
   // If originalConn does not, then TransposeConn shouldn't either.
   return PV_SUCCESS;
}

} // end namespace PV
