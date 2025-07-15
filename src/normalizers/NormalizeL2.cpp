/*
 * NormalizeL2.cpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#include "NormalizeL2.hpp"
#include "structures/Weights.hpp"

namespace PV {

NormalizeL2::NormalizeL2() { initialize_base(); }

NormalizeL2::NormalizeL2(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize_base();
   initialize(paramsIO, comm);
}

int NormalizeL2::initialize_base() { return PV_SUCCESS; }

void NormalizeL2::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   NormalizeMultiply::initialize(paramsIO, comm);
}

int NormalizeL2::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = NormalizeMultiply::ioParamsFillGroup(ioSwitch);
   ioParam_minL2NormTolerated(ioSwitch);
   return status;
}

void NormalizeL2::ioParam_minL2NormTolerated(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "minL2NormTolerated", &minL2NormTolerated);
}

int NormalizeL2::normalizeWeights() {
   int status = PV_SUCCESS;

   assert(!mWeightsList.empty());

   // All connections in the group must have the same values of sharedWeights, numArbors, and
   // numDataPatches
   Weights *weights0 = mWeightsList[0];

   float scaleFactor = 1.0f;
   if (mNormalizeFromPostPerspective) {
      if (!weights0->getSharedWeightsFlag()) {
         Fatal().printf(
               "NormalizeL2 error for %s: normalizeFromPostPerspective is true but connection does "
               "not use shared weights.\n",
               weights0->getName().c_str());
      }
      PVLayerLoc const &preLoc  = weights0->getGeometry()->getPreLoc();
      PVLayerLoc const &postLoc = weights0->getGeometry()->getPostLoc();
      int numNeuronsPre         = preLoc.nx * preLoc.ny * preLoc.nf;
      int numNeuronsPost        = postLoc.nx * postLoc.ny * postLoc.nf;
      scaleFactor               = ((float)numNeuronsPost) / ((float)numNeuronsPre);
   }
   scaleFactor *= mStrength;

   status = NormalizeMultiply::normalizeWeights(); // applies normalize_cutoff threshold and
   // rMinX,rMinY

   int nArbors        = weights0->getNumArbors();
   int numDataPatches = weights0->getNumDataPatches();
   
   std::vector<float> sumSquares(numDataPatches);
   if (mNormalizeArborsIndividually) {
      for (int arborID = 0; arborID < nArbors; arborID++) {
         std::fill(sumSquares.begin(), sumSquares.end(), 0.0f);
         for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
            float sumsq = 0.0f;
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weightsPerPatch   = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               accumulateSumSquared(dataStartPatch, weightsPerPatch, &sumsq);
            }
            sumSquares[patchindex] = sumsq;
         }
         if (mConnectionData->getPreIsBroadcast()) {
            pvAssert(mConnectionData->getPre()->getLayerLoc()->nx == 1);
            pvAssert(mConnectionData->getPre()->getLayerLoc()->ny == 1);
            MPI_Allreduce(
                  MPI_IN_PLACE,
                  sumSquares.data(),
                  numDataPatches,
                  MPI_FLOAT,
                  MPI_SUM,
                  mCommunicator->communicator());
         }
         for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
            float l2norm = std::sqrt(sumSquares[patchindex]);
            if (std::fabs(l2norm) <= minL2NormTolerated) {
               WarnLog().printf(
                     "for NormalizeL2 \"%s\": sum of squares of weights in patch %d of arbor %d is "
                     "within minL2NormTolerated=%f of zero.  Weights in this patch unchanged.\n",
                     getName(),
                     patchindex,
                     arborID,
                     (double)minL2NormTolerated);
               continue;
            }
            float normalizationFactor = scaleFactor / l2norm;
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weightsPerPatch   = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               normalizePatch(dataStartPatch, weightsPerPatch, normalizationFactor);
            }
         }
      }
   }
   else {
      for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
         float sumsq = 0.0f;
         for (int arborID = 0; arborID < nArbors; arborID++) {
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weightsPerPatch   = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               accumulateSumSquared(dataStartPatch, weightsPerPatch, &sumsq);
            }
         }
         sumSquares[patchindex] = sumsq;
      }
      if (mConnectionData->getPreIsBroadcast()) {
         pvAssert(mConnectionData->getPre()->getLayerLoc()->nx == 1);
         pvAssert(mConnectionData->getPre()->getLayerLoc()->ny == 1);
         MPI_Allreduce(
               MPI_IN_PLACE,
               sumSquares.data(),
               numDataPatches,
               MPI_FLOAT,
               MPI_SUM,
               mCommunicator->communicator());
      }
      for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
         float l2norm = std::sqrt(sumSquares[patchindex]);
         if (std::fabs(l2norm) <= minL2NormTolerated) {
            WarnLog().printf(
                  "for NormalizeL2 \"%s\": sum of squares of weights in patch %d is within "
                  "minL2NormTolerated=%f of zero.  Weights in this patch unchanged.\n",
                  getName(),
                  patchindex,
                  (double)minL2NormTolerated);
            continue;
         }
         float normalizationFactor = scaleFactor / l2norm;
         for (int arborID = 0; arborID < nArbors; arborID++) {
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weightsPerPatch   = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               normalizePatch(dataStartPatch, weightsPerPatch, normalizationFactor);
            }
         }
      }
   }
   return status;
}

NormalizeL2::~NormalizeL2() {}

} /* namespace PV */
