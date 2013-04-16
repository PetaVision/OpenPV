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
   params = NULL;
   return PV_SUCCESS;
}

// NormalizeBase does not directly call initialize since it is an abstract base class.
// Subclasses should call NormalizeBase::initialize from their own initialize routine
// This allows virtual methods called from initialize to be aware of which class's constructor was called.
int NormalizeBase::initialize(const char * name, PVParams * params) {
   // name is the name of a group in the PVParams object.  Parameters related to normalization should be in the indicated group.
   int status = PV_SUCCESS;
   if (params->group(name)) {
      this->name = strdup(name);
      this->params = params;
      setParams();
   }
   else {
      fprintf(stderr, "NormalizeBase error: group \"%s\" does not exist in given params file.\n", name);
      status = PV_FAILURE;
   }
   return status;
}

int NormalizeBase::setParams() {
   readStrength();
   readNormalizeCutoff();
   readSymmetrizeWeights();
   readNormalizeFromPostPerspective();
   readNormalizeArborsIndividually();
   return PV_SUCCESS;
}

int NormalizeBase::normalizeWeights(HyPerConn * conn) {
   int status = PV_SUCCESS;
#ifdef USE_SHMGET
#ifdef PV_USE_MPI
   if (shmget_flag && !shmget_owner[0]) { // Assumes that all arbors are owned by the same process
      MPI_Barrier(conn->getParent()->icCommunicator()->communicator());
   }
   return status;
#endif // PV_USE_MPI
#endif // USE_SHMGET
   if (symmetrizeWeightsFlag) {
      status = symmetrizeWeights(conn);
      if (status != PV_SUCCESS) return status;
   }

   if (normalize_cutoff>0) {
      int num_arbors = conn->numberOfAxonalArborLists();
      int num_patches = conn->getNumDataPatches();
      int num_weights_in_patch = conn->xPatchSize()*conn->yPatchSize()*conn->fPatchSize();
      if (normalizeArborsIndividually) {
         for (int arbor=0; arbor<num_arbors; arbor++) {
            pvdata_t * dataStart = conn->get_wDataStart(arbor);
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
               pvdata_t * dataStart = conn->get_wDataStart(arbor);
               accumulateMax(dataStart+patchindex*num_weights_in_patch, num_weights_in_patch, &max);
            }
            for (int arbor=0; arbor<num_arbors; arbor++) {
               pvdata_t * dataStart = conn->get_wDataStart(arbor);
               applyThreshold(dataStart+patchindex*num_weights_in_patch, num_weights_in_patch, max);
            }
         }
      }
   }
#ifdef USE_SHMGET
#ifdef PV_USE_MPI
   if (shmget_flag) {
      assert(shmget_owner[0]); // Assumes that all arbors are owned by the same process
      MPI_Barrier(conn->getParent()->icCommunicator()->communicator());
   }
#endif // PV_USE_MPI
#endif // USE_SHMGET
   return status;
}

int NormalizeBase::accumulateSum(pvdata_t * dataPatchStart, int weights_in_patch, double * sum) {
   // Do not call with sum uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over several patches with multiple calls
   for (int k=0; k<weights_in_patch; k++) {
      pvdata_t w = dataPatchStart[k];
      *sum += w;
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
                     pvdata_t * kerW = conn->get_wDataStart(arborID) + iKernel*nxp*nyp*nfp;
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
         pvdata_t * kerW = conn->get_wDataStart(arborID)+iKernel*nxp*nyp*nfp;
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

} // end namespace PV

