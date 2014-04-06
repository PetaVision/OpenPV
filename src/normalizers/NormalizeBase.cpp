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
}

int NormalizeBase::initialize_base() {
   name = NULL;
   callingConn = NULL;
   strength = 1.0f;
   rMinX = 0.0f;
   rMinY = 0.0f;
   normalize_cutoff = 0.0f;
   symmetrizeWeightsFlag = false;
   normalizeFromPostPerspective = false;
   normalizeArborsIndividually = false;
   return PV_SUCCESS;
}

// NormalizeBase does not directly call initialize since it is an abstract base class.
// Subclasses should call NormalizeBase::initialize from their own initialize routine
// This allows virtual methods called from initialize to be aware of which class's constructor was called.
int NormalizeBase::initialize(HyPerConn * callingConn) {
   // name is the name of a group in the PVParams object.  Parameters related to normalization should be in the indicated group.
   int status = PV_SUCCESS;
   this->name = strdup(callingConn->getName());
   this->callingConn = callingConn;
   return status;
}

int NormalizeBase::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_strength(ioFlag);
   ioParam_rMinX(ioFlag);
   ioParam_rMinY(ioFlag);
   ioParam_normalize_cutoff(ioFlag);
   ioParam_symmetrizeWeights(ioFlag);
   ioParam_normalizeFromPostPerspective(ioFlag);
   ioParam_normalizeArborsIndividually(ioFlag);
   return PV_SUCCESS;
}

void NormalizeBase::ioParam_strength(enum ParamsIOFlag ioFlag) {
   callingConn->ioParam_strength(ioFlag, &strength, true/*warnIfAbsent*/);
}

void NormalizeBase::ioParam_rMinX(enum ParamsIOFlag ioFlag) {
   parent()->ioParamValue(ioFlag, name, "rMinX", &rMinX, rMinX);
}

void NormalizeBase::ioParam_rMinY(enum ParamsIOFlag ioFlag) {
   parent()->ioParamValue(ioFlag, name, "rMinY", &rMinY, rMinY);
}

void NormalizeBase::ioParam_normalize_cutoff(enum ParamsIOFlag ioFlag) {
   parent()->ioParamValue(ioFlag, name, "normalize_cutoff", &normalize_cutoff, normalize_cutoff);
}

void NormalizeBase::ioParam_symmetrizeWeights(enum ParamsIOFlag ioFlag) {
   parent()->ioParamValue(ioFlag, name, "symmetrizeWeights", &symmetrizeWeightsFlag, false);
}

void NormalizeBase::ioParam_normalizeFromPostPerspective(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && !parent()->parameters()->present(name, "normalizeFromPostPerspective") && parent()->parameters()->present(name, "normalize_arbors_individually")) {
      if (parent()->columnId()==0) {
         fprintf(stderr, "Normalizer \"%s\": parameter name normalizeTotalToPost is deprecated.  Use normalizeFromPostPerspective.\n", name);
      }
      normalizeFromPostPerspective = parent()->parameters()->value(name, "normalizeTotalToPost");
      return;
   }
   parent()->ioParamValue(ioFlag, name, "normalizeFromPostPerspective", &normalizeFromPostPerspective, false/*default value*/, true/*warnIfAbsent*/);
}

void NormalizeBase::ioParam_normalizeArborsIndividually(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && !parent()->parameters()->present(name, "normalizeArborsIndividually") && parent()->parameters()->present(name, "normalize_arbors_individually")) {
      if (parent()->columnId()==0) {
         fprintf(stderr, "Normalizer \"%s\": parameter name normalize_arbors_individually is deprecated.  Use normalizeArborsIndividually.\n", name);
      }
      normalizeArborsIndividually = parent()->parameters()->value(name, "normalize_arbors_individually");
      return;
   }
   parent()->ioParamValue(ioFlag, name, "normalizeArborsIndividually", &normalizeArborsIndividually, false/*default*/, true/*warnIfAbsent*/);
}

int NormalizeBase::normalizeWeights(HyPerConn * conn) {
   int status = PV_SUCCESS;
#ifdef USE_SHMGET
#ifdef PV_USE_MPI
   if (conn->getShmgetFlag()) assert(conn->getShmgetOwner(0)); // Only called by subclasses of normalizeWeights, and if shmgetFlag is set, only by the owner
#endif // PV_USE_MPI
#endif // USE_SHMGET
   if (symmetrizeWeightsFlag) {
      status = symmetrizeWeights(conn);
      if (status != PV_SUCCESS) return status;
   }

   if (rMinX > 0.5f && rMinY > 0.5f){
	   int num_arbors = conn->numberOfAxonalArborLists();
	   int num_patches = conn->getNumDataPatches();
	   int num_weights_in_patch = conn->xPatchSize()*conn->yPatchSize()*conn->fPatchSize();
       for (int arbor=0; arbor<num_arbors; arbor++) {
          pvwdata_t * dataPatchStart = conn->get_wDataStart(arbor);
          for (int patchindex=0; patchindex<num_patches; patchindex++) {
        	  applyRMin(dataPatchStart+patchindex*num_weights_in_patch, rMinX, rMinY,
        		  conn->xPatchSize(), conn->yPatchSize(), conn->xPatchStride(), conn->yPatchStride());
          }
       }
   }
   if (normalize_cutoff>0) {
      int num_arbors = conn->numberOfAxonalArborLists();
      int num_patches = conn->getNumDataPatches();
      int num_weights_in_patch = conn->xPatchSize()*conn->yPatchSize()*conn->fPatchSize();
      if (normalizeArborsIndividually) {
         for (int arbor=0; arbor<num_arbors; arbor++) {
            pvwdata_t * dataStart = conn->get_wDataStart(arbor);
            float max = 0.0f;
            for (int patchindex=0; patchindex<num_patches; patchindex++) {
               accumulateMax(dataStart+patchindex*num_weights_in_patch, num_weights_in_patch, &max);
            }
            for (int patchindex=0; patchindex<num_patches; patchindex++) {
               applyThreshold(dataStart+patchindex*num_weights_in_patch, num_weights_in_patch, max);
            }
         }
      }
      else {
         for (int patchindex=0; patchindex<num_patches; patchindex++) {
            float max = 0.0f;
            for (int arbor=0; arbor<num_arbors; arbor++) {
               pvwdata_t * dataStart = conn->get_wDataStart(arbor);
               accumulateMax(dataStart+patchindex*num_weights_in_patch, num_weights_in_patch, &max);
            }
            for (int arbor=0; arbor<num_arbors; arbor++) {
               pvwdata_t * dataStart = conn->get_wDataStart(arbor);
               applyThreshold(dataStart+patchindex*num_weights_in_patch, num_weights_in_patch, max);
            }
         }
      }
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

int NormalizeBase::accumulateSumShrunken(pvdata_t * dataPatchStart, double * sum,
		int nxpShrunken, int nypShrunken, int offsetShrunken, int xPatchStride, int yPatchStride) {
   // Do not call with sumsq uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over several patches with multiple calls
	pvdata_t * dataPatchStartOffset = dataPatchStart + offsetShrunken;
	int weights_in_row = xPatchStride * nxpShrunken;
	for (int ky = 0; ky<nypShrunken; ky++){
		for (int k=0; k<weights_in_row; k++) {
			pvdata_t w = dataPatchStartOffset[k];
			*sum += w;
		}
		dataPatchStartOffset += yPatchStride;
	}
   return PV_SUCCESS;
}

int NormalizeBase::accumulateSumSquared(pvdata_t * dataPatchStart, int weights_in_patch, double * sumsq) {
   // Do not call with sumsq uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over several patches with multiple calls
   for (int k=0; k<weights_in_patch; k++) {
      pvdata_t w = dataPatchStart[k];
      *sumsq += w*w;
   }
   return PV_SUCCESS;
}

int NormalizeBase::accumulateSumSquaredShrunken(pvdata_t * dataPatchStart, double * sumsq,
		int nxpShrunken, int nypShrunken, int offsetShrunken, int xPatchStride, int yPatchStride) {
   // Do not call with sumsq uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over several patches with multiple calls
	pvdata_t * dataPatchStartOffset = dataPatchStart + offsetShrunken;
	int weights_in_row = xPatchStride * nxpShrunken;
	for (int ky = 0; ky<nypShrunken; ky++){
		for (int k=0; k<weights_in_row; k++) {
			pvdata_t w = dataPatchStartOffset[k];
			*sumsq += w*w;
		}
		dataPatchStartOffset += yPatchStride;
	}
   return PV_SUCCESS;
}

int NormalizeBase::accumulateMax(pvdata_t * dataPatchStart, int weights_in_patch, float * max) {
   // Do not call with max uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over several patches with multiple calls
   for (int k=0; k<weights_in_patch; k++) {
      pvdata_t w = dataPatchStart[k];
      if (w>*max) *max=w;
   }
   return PV_SUCCESS;
}

int NormalizeBase::applyThreshold(pvdata_t * dataPatchStart, int weights_in_patch, float wMax) {
   assert(normalize_cutoff>0); // Don't call this routine unless normalize_cutoff was set
   float threshold = wMax * normalize_cutoff;
   for (int k=0; k<weights_in_patch; k++) {
      if (fabsf(dataPatchStart[k])<threshold) dataPatchStart[k] = 0;
   }
   return PV_SUCCESS;
}

// dataPatchStart points to head of full-sized patch
// rMinX, rMinY are the minimum radii from the center of the patch,
// all weights inside (non-inclusive) of this radius are set to zero
// the diameter of the central exclusion region is truncated to the nearest integer value, which may be zero
int NormalizeBase::applyRMin(pvdata_t * dataPatchStart, float rMinX, float rMinY,
		int nxp, int nyp, int xPatchStride, int yPatchStride) {
	if(rMinX==0 && rMinY == 0) return PV_SUCCESS;
	int fullWidthX = floor(2 * rMinX);
	int fullWidthY = floor(2 * rMinY);
	int offsetX = ceil((nxp - fullWidthX) / 2.0);
	int offsetY = ceil((nyp - fullWidthY) / 2.0);
	int widthX = nxp - 2 * offsetX;
	int widthY = nyp - 2 * offsetY;
	pvdata_t * rMinPatchStart = dataPatchStart + offsetY * yPatchStride + offsetX * xPatchStride;
	int weights_in_row = xPatchStride * widthX;
	for (int ky = 0; ky<widthY; ky++){
		for (int k=0; k<weights_in_row; k++) {
			rMinPatchStart[k] = 0;
		}
		rMinPatchStart += yPatchStride;
	}
  return PV_SUCCESS;
}

int NormalizeBase::symmetrizeWeights(HyPerConn * conn) {
   assert(symmetrizeWeightsFlag); // Don't call this routine unless symmetrizeWeights was set
   int status = PV_SUCCESS;
   KernelConn * kconn = dynamic_cast<KernelConn *>(conn);
   if (!kconn) {
      fprintf(stderr, "NormalizeSum error for connection \"%s\": symmetrizeWeights is true but connection is not a KernelConn\n", conn->getName());
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

void NormalizeBase::normalizePatch(pvdata_t * dataStartPatch, int weights_per_patch, float multiplier) {
   for (int k=0; k<weights_per_patch; k++) dataStartPatch[k] *= multiplier;
}

HyPerCol * NormalizeBase::parent() {
   return callingConn->getParent();
}

} // end namespace PV

