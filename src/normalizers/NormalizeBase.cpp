/*
 * NormalizeBase.cpp
 *
 *  Created on: Apr 5, 2013
 *      Author: pschultz
 */

#include "NormalizeBase.hpp"

namespace PV {

NormalizeBase::NormalizeBase() {
   initialize_base();
}

NormalizeBase::~NormalizeBase() {
   free(name);
   free(connectionList); // normalizer does not own the individual connections in the list, so don't free them
}

int NormalizeBase::initialize_base() {
   name = NULL;
   parentHyPerCol = NULL;
   connectionList = NULL;
   numConnections = 0;
   strength = 1.0f;
   // normalizeFromPostPerspective,rMinX,rMinY,normalize_cutoff moved to NormalizeMultiply
#ifdef OBSOLETE // Marked obsolete Oct 24, 2014.  symmetrizeWeights is too specialized for NormalizeBase.  Create a new subclass if you want to restore this functionality
   symmetrizeWeightsFlag = false;
#endif // OBSOLETE
   normalizeArborsIndividually = false;
   normalizeOnInitialize = true;
   normalizeOnWeightUpdate = true;
   return PV_SUCCESS;
}

// NormalizeBase does not directly call initialize since it is an abstract base class.
// Subclasses should call NormalizeBase::initialize from their own initialize routine
// This allows virtual methods called from initialize to be aware of which class's constructor was called.
int NormalizeBase::initialize(const char * name, HyPerCol * hc) {
   // name is the name of a group in the PVParams object.  Parameters related to normalization should be in the indicated group.

   int status = PV_SUCCESS;
   this->connectionList = NULL;
   this->numConnections = 0;
   this->name = strdup(name);
   if (this->name==NULL) {
      fprintf(stderr, "Rank %d error: unable to allocate memory for name \"%s\" of normalizer object.\n", hc->columnId(), name);
      exit(EXIT_FAILURE);
   }
   this->parentHyPerCol = hc;
   status = hc->addNormalizer(this);
   return status;
}

int NormalizeBase::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
      ioParam_strength(ioFlag);
   // normalizeFromPostPerspective,rMinX,rMinY,normalize_cutoff moved to NormalizeMultiply
#ifdef OBSOLETE // Marked obsolete Oct 24, 2014.  symmetrizeWeights is too specialized for NormalizeBase.  Create a new subclass to restore this functionality
   ioParam_symmetrizeWeights(ioFlag);
#endif // OBSOLETE
   ioParam_normalizeArborsIndividually(ioFlag);
   ioParam_normalizeOnInitialize(ioFlag);
   ioParam_normalizeOnWeightUpdate(ioFlag);
   return PV_SUCCESS;
}

void NormalizeBase::ioParam_strength(enum ParamsIOFlag ioFlag) {
   parent()->ioParamValue(ioFlag, name, "strength", &strength, strength/*default*/, true/*warn if absent*/);
}

// normalizeFromPostPerspective,rMinX,rMinY,normalize_cutoff moved to NormalizeMultiply

#ifdef OBSOLETE // Marked obsolete Oct 24, 2014.  symmetrizeWeights is too specialized for NormalizeBase.  Create a new subclass to restore this functionality
void NormalizeBase::ioParam_symmetrizeWeights(enum ParamsIOFlag ioFlag) {
   parent()->ioParamValue(ioFlag, name, "symmetrizeWeights", &symmetrizeWeightsFlag, false);
}
#endif // OBSOLETE

void NormalizeBase::ioParam_normalizeArborsIndividually(enum ParamsIOFlag ioFlag) {
   // normalize_arbors_individually as a parameter name was deprecated April 19, 2013 and marked obsolete October 24, 2014
   if (ioFlag==PARAMS_IO_READ && !parent()->parameters()->present(name, "normalizeArborsIndividually") && parent()->parameters()->present(name, "normalize_arbors_individually")) {
      if (parent()->columnId()==0) {
         fprintf(stderr, "Normalizer \"%s\": parameter name normalize_arbors_individually is obsolete.  Use normalizeArborsIndividually.\n", name);
      }
      MPI_Barrier(parent()->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   parent()->ioParamValue(ioFlag, name, "normalizeArborsIndividually", &normalizeArborsIndividually, false/*default*/, true/*warnIfAbsent*/);
}

void NormalizeBase::ioParam_normalizeOnInitialize(enum ParamsIOFlag ioFlag) {
   parent()->ioParamValue(ioFlag, name, "normalizeOnInitialize", &normalizeOnInitialize, normalizeOnInitialize);
}

void NormalizeBase::ioParam_normalizeOnWeightUpdate(enum ParamsIOFlag ioFlag) {
   parent()->ioParamValue(ioFlag, name, "normalizeOnWeightUpdate", &normalizeOnWeightUpdate, normalizeOnWeightUpdate);
}


int NormalizeBase::normalizeWeightsWrapper() {
   int status = PV_SUCCESS;
   HyPerConn * callingConn = connectionList[0];
   // TODO: For groups, how should we enforce groups of connections, each with its own lastUpdateTime?
   double simTime = callingConn->getParent()->simulationTime();
   if ( (normalizeOnInitialize && simTime == callingConn->getParent()->getStartTime()) ||
        (normalizeOnWeightUpdate && simTime == callingConn->getLastUpdateTime()) ) {
      status = normalizeWeights();
   }
   return status;
}

int NormalizeBase::normalizeWeights() {
   assert((normalizeOnInitialize && connectionList[0]->getParent()->simulationTime() == connectionList[0]->getParent()->getStartTime()) ||
          (normalizeOnWeightUpdate && connectionList[0]->getParent()->simulationTime() == connectionList[0]->getLastUpdateTime()));
   int status = PV_SUCCESS;
#ifdef OBSOLETE // Marked obsolete Dec 9, 2014
#ifdef USE_SHMGET
#ifdef PV_USE_MPI
   if (conn->getShmgetFlag()) assert(conn->getShmgetOwner(0)); // Only called by subclasses of normalizeWeights, and if shmgetFlag is set, only by the owner
#endif // PV_USE_MPI
#endif // USE_SHMGET
#endif // OBSOLETE
   for (int c=0; c<numConnections; c++) {
      HyPerConn * conn = connectionList[c];
#ifdef OBSOLETE // Marked obsolete Oct 24, 2014.  symmetrizeWeights is too specialized for NormalizeBase.  Create a new subclass to restore this functionality
      if (symmetrizeWeightsFlag) {
         status = symmetrizeWeights(conn);
         if (status != PV_SUCCESS) return status;
      }
#endif //OBSOLETE

      // normalizeFromPostPerspective,rMinX,rMinY,normalize_cutoff moved to NormalizeMultiply
   }
   return status;
}

int NormalizeBase::accumulateSum(pvwdata_t * dataPatchStart, int weights_in_patch, double * sum) {
   // Do not call with sum uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over several patches with multiple calls
   for (int k=0; k<weights_in_patch; k++) {
      pvwdata_t w = dataPatchStart[k];
      //TODO-CER-2014.4.4 - weight conversion
      *sum += w;
   }
   return PV_SUCCESS;
}

int NormalizeBase::accumulateSumShrunken(pvwdata_t * dataPatchStart, double * sum,
		int nxpShrunken, int nypShrunken, int offsetShrunken, int xPatchStride, int yPatchStride) {
   // Do not call with sumsq uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over several patches with multiple calls
	pvwdata_t * dataPatchStartOffset = dataPatchStart + offsetShrunken;
	int weights_in_row = xPatchStride * nxpShrunken;
	for (int ky = 0; ky<nypShrunken; ky++){
		for (int k=0; k<weights_in_row; k++) {
			pvwdata_t w = dataPatchStartOffset[k];
			*sum += w;
		}
		dataPatchStartOffset += yPatchStride;
	}
   return PV_SUCCESS;
}

int NormalizeBase::accumulateSumSquared(pvwdata_t * dataPatchStart, int weights_in_patch, double * sumsq) {
   // Do not call with sumsq uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over several patches with multiple calls
   for (int k=0; k<weights_in_patch; k++) {
      pvwdata_t w = dataPatchStart[k];
      *sumsq += w*w;
   }
   return PV_SUCCESS;
}

int NormalizeBase::accumulateSumSquaredShrunken(pvwdata_t * dataPatchStart, double * sumsq,
		int nxpShrunken, int nypShrunken, int offsetShrunken, int xPatchStride, int yPatchStride) {
   // Do not call with sumsq uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over several patches with multiple calls
	pvwdata_t * dataPatchStartOffset = dataPatchStart + offsetShrunken;
	int weights_in_row = xPatchStride * nxpShrunken;
	for (int ky = 0; ky<nypShrunken; ky++){
		for (int k=0; k<weights_in_row; k++) {
			pvwdata_t w = dataPatchStartOffset[k];
			*sumsq += w*w;
		}
		dataPatchStartOffset += yPatchStride;
	}
   return PV_SUCCESS;
}

int NormalizeBase::accumulateMaxAbs(pvwdata_t * dataPatchStart, int weights_in_patch, float * max) {
   // Do not call with max uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over several patches with multiple calls
   float newmax = *max;
   for (int k=0; k<weights_in_patch; k++) {
      pvwdata_t w = fabsf(dataPatchStart[k]);
      if (w>newmax) newmax=w;
   }
   *max = newmax;
   return PV_SUCCESS;
}

int NormalizeBase::accumulateMax(pvwdata_t * dataPatchStart, int weights_in_patch, float * max) {
   // Do not call with max uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over several patches with multiple calls
   float newmax = *max;
   for (int k=0; k<weights_in_patch; k++) {
      pvwdata_t w = dataPatchStart[k];
      if (w>newmax) newmax=w;
   }
   *max = newmax;
   return PV_SUCCESS;
}

int NormalizeBase::accumulateMin(pvwdata_t * dataPatchStart, int weights_in_patch, float * min) {
   // Do not call with min uninitialized.
   // min is cleared inside this routine so that you can accumulate the stats over several patches with multiple calls
   float newmin = *min;
   for (int k=0; k<weights_in_patch; k++) {
      pvwdata_t w = dataPatchStart[k];
      if (w<newmin) newmin=w;
   }
   *min = newmin;
   return PV_SUCCESS;
}

// normalizeFromPostPerspective,rMinX,rMinY,normalize_cutoff moved to NormalizeMultiply

#ifdef OBSOLETE // Marked obsolete Oct 24, 2014.  symmetrizeWeights is too specialized for NormalizeBase.  Create a new subclass to restore this functionality
int NormalizeBase::symmetrizeWeights(HyPerConn * conn) {
   assert(symmetrizeWeightsFlag); // Don't call this routine unless symmetrizeWeights was set
   int status = PV_SUCCESS;
   if (conn->usingSharedWeights()==false) {
      fprintf(stderr, "NormalizeSum error for connection \"%s\": symmetrizeWeights is true but connection does not use shared weights\n", conn->getName());
      exit(EXIT_FAILURE);
   }
   HyPerLayer * pre = conn->preSynapticLayer();
   HyPerLayer * post = conn->postSynapticLayer();
   int nxPre = pre->getLayerLoc()->nx;
   int nxPost = post->getLayerLoc()->nx;
   int nyPre = pre->getLayerLoc()->ny;
   int nyPost = post->getLayerLoc()->ny;
   int nfPre = pre->getLayerLoc()->nf;
   int nfPost = post->getLayerLoc()->nf;
   if (nxPre != nxPost || nyPre != nyPost || nfPre != nfPost) {
      fprintf(stderr, "NormalizeSum error for connection \"%s\": symmetrizeWeights is true but pre layer \"%s\" and post layer \"%s\" have different dimensions (%d,%d,%d) versus (%d,%d,%d)\n",
            conn->getName(), pre->getName(), post->getName(), nxPre,nyPre,nfPre, nxPost,nyPost,nfPost);
      exit(EXIT_FAILURE);
   }

   int nxp = conn->xPatchSize();
   int nyp = conn->yPatchSize();
   int nfp = conn->fPatchSize();
   int numPatches = conn->getNumDataPatches();
   pvdata_t * symPatches = (pvdata_t *) calloc(nxp*nyp*nfp*numPatches, sizeof(pvdata_t));
   if(symPatches == NULL) {
      fprintf(stderr, "symmetrizeWeights error for connection \"%s\": unable to allocate memory for symmetrizing weights\n", conn->getName());
   }

   const int sy = nxp * nfp;
   const float deltaTheta = PI / nfp;
   const float offsetTheta = 0.5f * deltaTheta;
   const int kyMid = nyp / 2;
   const int kxMid = nxp / 2;

   for (int arborID=0; arborID<conn->numberOfAxonalArborLists(); arborID++) {
      for (int iSymKernel = 0; iSymKernel < numPatches; iSymKernel++) {
         pvdata_t * symW = symPatches + iSymKernel*nxp*nyp*nfp;
         float symTheta = offsetTheta + iSymKernel * deltaTheta;
         for (int kySym = 0; kySym < nyp; kySym++) {
            float dySym = kySym - kyMid;
            for (int kxSym = 0; kxSym < nxp; kxSym++) {
               float dxSym = kxSym - kxMid;
               float distSym = sqrt(dxSym * dxSym + dySym * dySym);
               if (distSym > abs(kxMid > kyMid ? kxMid : kyMid)) {
                  continue;
               }
               float dyPrime = dySym * cos(symTheta) - dxSym * sin(symTheta);
               float dxPrime = dxSym * cos(symTheta) + dySym * sin(symTheta);
               for (int kfSym = 0; kfSym < nfp; kfSym++) {
                  int kDf = kfSym - iSymKernel;
                  int iSymW = kfSym + nfp * kxSym + sy * kySym;
                  for (int iKernel = 0; iKernel < nfp; iKernel++) {
                     pvwdata_t * kerW = conn->get_wDataStart(arborID) + iKernel*nxp*nyp*nfp;
                     int kfRot = iKernel + kDf;
                     if (kfRot < 0) {
                        kfRot = nfp + kfRot;
                     }
                     else {
                        kfRot = kfRot % nfp;
                     }
                     float rotTheta = offsetTheta + iKernel * deltaTheta;
                     float yRot = dyPrime * cos(rotTheta) + dxPrime * sin(rotTheta);
                     float xRot = dxPrime * cos(rotTheta) - dyPrime * sin(rotTheta);
                     yRot += kyMid;
                     xRot += kxMid;
                     // should find nearest neighbors and do weighted average
                     int kyRot = yRot + 0.5f;
                     int kxRot = xRot + 0.5f;
                     int iRotW = kfRot + nfp * kxRot + sy * kyRot;
                     symW[iSymW] += kerW[iRotW] / nfp;
                  } // kfRot
               } // kfSymm
            } // kxSym
         } // kySym
      } // iKernel
      const int num_weights = nfp * nxp * nyp;
      for (int iKernel = 0; iKernel < numPatches; iKernel++) {
         pvwdata_t * kerW = conn->get_wDataStart(arborID)+iKernel*nxp*nyp*nfp;
         pvdata_t * symW = symPatches + iKernel*nxp*nyp*nfp;
         for (int iW = 0; iW < num_weights; iW++) {
            kerW[iW] = symW[iW];
         }
      } // iKernel

   }

   free(symPatches);

   return status;
}
#endif // OBSOLETE

int NormalizeBase::addConnToList(HyPerConn * newConn) {
   HyPerConn ** newList = NULL;
   if (connectionList) {
      newList = (HyPerConn **) realloc(connectionList, sizeof(*connectionList)*(numConnections+1));
   }
   else {
      newList = (HyPerConn **) malloc(sizeof(*connectionList)*(numConnections+1));
   }
   if (newList==NULL) {
      fprintf(stderr, "Normalizer \"%s\" unable to add connection \"%s\" as connection number %d : %s\n", name, newConn->getName(), numConnections+1, strerror(errno));
      exit(EXIT_FAILURE);
   }
   connectionList = newList;
   connectionList[numConnections] = newConn;
   numConnections++;
   if (parent()->columnId()==0) {
      printf("Adding connection \"%s\" to normalizer group \"%s\".\n", newConn->getName(), this->getName());
   }
   return PV_SUCCESS;
}

void NormalizeBase::normalizePatch(pvwdata_t * dataStartPatch, int weights_per_patch, float multiplier) {
   for (int k=0; k<weights_per_patch; k++) dataStartPatch[k] *= multiplier;
}

} // end namespace PV

