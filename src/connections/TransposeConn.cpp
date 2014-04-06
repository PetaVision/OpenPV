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

TransposeConn::TransposeConn(const char * name, HyPerCol * hc) {
   initialize_base();
   int status = initialize(name, hc);
}

TransposeConn::~TransposeConn() {
   free(originalConnName); originalConnName = NULL;
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

int TransposeConn::initialize(const char * name, HyPerCol * hc) {
   int status = PV_SUCCESS;
   if (status == PV_SUCCESS) status = KernelConn::initialize(name, hc);
   return status;
}

int TransposeConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = KernelConn::ioParamsFillGroup(ioFlag);
   ioParam_originalConnName(ioFlag);
   return status;
}

// We override many ioParam-methods because TransposeConn will determine
// the associated parameters from the originalConn's values.
// communicateInitInfo will check if those parameters exist in params for
// the CloneKernelConn group, and whether they are consistent with the
// originalConn parameters.
// If consistent, issue a warning that the param is unnecessary and continue.
// If inconsistent, issue an error and quit.
// We can't do that in the read-method because we can't be sure originalConn
// has set its own parameter yet (or even if it's been instantiated),
// and in theory originalConn could be a subclass that determines
// the parameter some way other than reading its own parameter
// group's param directly.

void TransposeConn::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   // TransposeConn doesn't use a weight initializer
   if (ioFlag==PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "weightInitType", NULL);
   }
}

void TransposeConn::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {
   // During the communication phase, numAxonalArbors will be copied from originalConn
}

void TransposeConn::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   // During the communication phase, plasticityFlag will be copied from originalConn
}

void TransposeConn::ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      combine_dW_with_W_flag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "combine_dW_with_W_flag", combine_dW_with_W_flag);
   }
}

void TransposeConn::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   // TransposeConn determines nxp from originalConn, during communicateInitInfo
}

void TransposeConn::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   // TransposeConn determines nyp from originalConn, during communicateInitInfo
}

void TransposeConn::ioParam_nxpShrunken(enum ParamsIOFlag ioFlag) {
   // TransposeConn doesn't use nxpShrunken
}

void TransposeConn::ioParam_nypShrunken(enum ParamsIOFlag ioFlag) {
   // TransposeConn doesn't use nypShrunken
}

void TransposeConn::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   // TransposeConn determines nfp from originalConn, during communicateInitInfo
}

void TransposeConn::ioParam_dWMax(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      dWMax = 1.0;
      parent->parameters()->handleUnnecessaryParameter(name, "dWMax", dWMax);
   }
}

void TransposeConn::ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      keepKernelsSynchronized_flag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "keepKernelsSynchronized", keepKernelsSynchronized_flag);
   }
}

void TransposeConn::ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      weightUpdatePeriod = parent->getDeltaTime();  // Ensures that every timestep updateState calls updateWeights, which will compare lastUpdateTime to originalConn's lastUpdateTime
      parent->parameters()->handleUnnecessaryParameter(name, "weightUpdatePeriod", weightUpdatePeriod);
   }
}

void TransposeConn::ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initialWeightUpdateTime = parent->getStartTime();
      parent->parameters()->handleUnnecessaryParameter(name, "initialWeightUpdateTime", initialWeightUpdateTime);
      weightUpdateTime = initialWeightUpdateTime;
   }
}

void TransposeConn::ioParam_useWindowPost(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      useWindowPost = false;
      parent->parameters()->handleUnnecessaryParameter(name, "useWindowPost");
   }
}

void TransposeConn::ioParam_shrinkPatches(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      shrinkPatches_flag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "shrinkPatches", shrinkPatches_flag);
   }
}

void TransposeConn::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      normalizer = NULL;
      normalizeMethod = strdup("none");
      parent->parameters()->handleUnnecessaryStringParameter(name, "normalizeMethod", "none");
   }
}

void TransposeConn::ioParam_originalConnName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "originalConnName", &originalConnName);
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

   if (!originalConn->getInitInfoCommunicatedFlag()) {
      if (parent->columnId()==0) {
         const char * connectiontype = parent->parameters()->groupKeywordFromName(name);
         printf("%s \"%s\" must wait until original connection \"%s\" has finished its communicateInitInfo stage.\n", connectiontype, name, originalConn->getName());
      }
      return PV_POSTPONE;
   }

   status = KernelConn::communicateInitInfo();
   if (status != PV_SUCCESS) return status;

   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * origPostLoc = originalConn->postSynapticLayer()->getLayerLoc();
   if (preLoc->nx != origPostLoc->nx || preLoc->ny != origPostLoc->ny || preLoc->nf != origPostLoc->nf) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: transpose's pre layer and original connection's post layer must have the same dimensions.\n", parent->parameters()->groupKeywordFromName(name), name);
         fprintf(stderr, "    (x=%d, y=%d, f=%d) versus (x=%d, y=%d, f=%d).\n", preLoc->nx, preLoc->ny, preLoc->nf, origPostLoc->nx, origPostLoc->ny, origPostLoc->nf);
      }
#if PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
   const PVLayerLoc * postLoc = pre->getLayerLoc();
   const PVLayerLoc * origPreLoc = originalConn->postSynapticLayer()->getLayerLoc();
   if (postLoc->nx != origPreLoc->nx || postLoc->ny != origPreLoc->ny || postLoc->nf != origPreLoc->nf) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: transpose's post layer and original connection's pre layer must have the same dimensions.\n", parent->parameters()->groupKeywordFromName(name), name);
         fprintf(stderr, "    (x=%d, y=%d, f=%d) versus (x=%d, y=%d, f=%d).\n", postLoc->nx, postLoc->ny, postLoc->nf, origPreLoc->nx, origPreLoc->ny, origPreLoc->nf);
      }
#if PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }


   numAxonalArborLists = originalConn->numberOfAxonalArborLists();
   parent->parameters()->handleUnnecessaryParameter(name, "numAxonalArbors", numAxonalArborLists);

   plasticityFlag = originalConn->getPlasticityFlag();
   parent->parameters()->handleUnnecessaryParameter(name, "plasticityFlag", plasticityFlag);

   if(originalConn->getShrinkPatches_flag()) {
      if (parent->columnId()==0) {
         fprintf(stderr, "TransposeConn \"%s\" error: original conn \"%s\" has shrinkPatches set to true.  TransposeConn has not been implemented for that case.\n", name, originalConn->getName());
      }
#if PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
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
   int nyp_orig = originalConn->yPatchSize();
   nxp = nxp_orig;
   if(xscaleDiff > 0 ) {
      nxp *= (int) pow( 2, xscaleDiff );
   }
   else if(xscaleDiff < 0) {
      nxp /= (int) pow(2,-xscaleDiff);
      assert(nxp_orig==nxp*pow( 2, (float) (-xscaleDiff) ));
   }

   int yscaleDiff = pre->getYScale() - post->getYScale();
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
   parent->parameters()->handleUnnecessaryParameter(name, "nxp", nxp);
   parent->parameters()->handleUnnecessaryParameter(name, "nyp", nyp);
   parent->parameters()->handleUnnecessaryParameter(name, "nxpShrunken", nxpShrunken);
   parent->parameters()->handleUnnecessaryParameter(name, "nypShrunken", nypShrunken);
   parent->parameters()->handleUnnecessaryParameter(name, "nfp", nfp);
   return PV_SUCCESS;

}

int TransposeConn::allocateDataStructures() {
   if (!originalConn->getDataStructuresAllocatedFlag()) {
      if (parent->columnId()==0) {
         const char * connectiontype = parent->parameters()->groupKeywordFromName(name);
         printf("%s \"%s\" must wait until original connection \"%s\" has finished its allocateDataStructures stage.\n", connectiontype, name, originalConn->getName());
      }
      return PV_POSTPONE;
   }

   KernelConn::allocateDataStructures();
   normalizer = NULL;
   // normalize_flag = false; // replaced by testing whether normalizer!=NULL
   return PV_SUCCESS;
}

PVPatch*** TransposeConn::initializeWeights(PVPatch*** patches, pvwdata_t** dataStart,
      int numPatches) {
   assert(originalConn->getDataStructuresAllocatedFlag()); // originalConn->dataStructurenAllocatedFlag checked in TransposeConn::allocateDataStructures()
   for (int arbor=0; arbor<numAxonalArborLists; arbor++) {
      transposeKernels(arbor);
   }
   return patches;
}


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
         pvwdata_t * dataStartFB = get_wDataHead(arborId, kernelnumberFB);
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
                  pvwdata_t * dataStartFF = originalConn->get_wDataHead(arborId, kernelnumberFF);
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
         pvwdata_t * dataStartFB = get_wDataHead(arborId, kernelnumberFB);
         int nxFB = nxp; // kpFB->nx;
         int nyFB = nyp; // kpFB->ny;
         int nfFB = nfp;
         for( int kyFB = 0; kyFB < nyFB; kyFB++ ) {
            int precelloffsety = kyFB % yscaleq;
            for( int kxFB = 0; kxFB < nxFB; kxFB++ ) {
               int precelloffsetx = kxFB % xscaleq;
               for( int kfFB = 0; kfFB < nfFB; kfFB++ ) {
                  int kernelnumberFF = (precelloffsety*xscaleq + precelloffsetx)*nfFB + kfFB;
                  pvwdata_t * dataStartFF = originalConn->get_wDataHead(arborId, kernelnumberFF);
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
