/*
 * NormalizeMultiply.cpp
 *
 *  Created on: Oct 24, 2014
 *      Author: pschultz
 */

#include "NormalizeMultiply.hpp"
#include "components/StrengthParam.hpp"
#include "structures/Weights.hpp"

namespace PV {

NormalizeMultiply::NormalizeMultiply(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

NormalizeMultiply::NormalizeMultiply() {}

NormalizeMultiply::~NormalizeMultiply() {}

void NormalizeMultiply::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   NormalizeBase::initialize(paramsIO, comm);
}

int NormalizeMultiply::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = NormalizeBase::ioParamsFillGroup(ioSwitch);
   ioParam_rMinX(ioSwitch);
   ioParam_rMinY(ioSwitch);
   ioParam_nonnegativeConstraintFlag(ioSwitch);
   ioParam_normalize_cutoff(ioSwitch);
   ioParam_normalizeFromPostPerspective(ioSwitch);
   return status;
}

void NormalizeMultiply::ioParam_rMinX(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "rMinX", &mRMinX);
}

void NormalizeMultiply::ioParam_rMinY(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "rMinY", &mRMinY);
}

void NormalizeMultiply::ioParam_nonnegativeConstraintFlag(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "nonnegativeConstraintFlag", &mNonnegativeConstraintFlag);
}

void NormalizeMultiply::ioParam_normalize_cutoff(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "normalize_cutoff", &mNormalizeCutoff);
}

void NormalizeMultiply::ioParam_normalizeFromPostPerspective(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "normalizeFromPostPerspective", &mNormalizeFromPostPerspective);
}

StrengthParam *NormalizeMultiply::retrieveStrengthParamIfNeeded(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *strengthParam = StrengthParam::ensureExists(
         message, getParamsIO(), mCommunicator);
   return strengthParam;
}

int NormalizeMultiply::normalizeWeights() {
   int status = PV_SUCCESS;

   // All connections in the group must have the same values of sharedWeights, numArbors, and
   // numDataPatches
   Weights *weights0 = mWeightsList[0];
   bool weights0Shared = weights0->getSharedWeightsFlag();
   for (auto &weights : mWeightsList) {
      bool weightsShared = weights->getSharedWeightsFlag();
      if (weightsShared != weights0Shared) {
         if (mCommunicator->globalCommRank() == 0) {
            ErrorLog().printf(
                  "%s: All connections in the normalization group must have the same sharedWeights "
                  "(%s has %d; %s has %d).\n",
                  this->getDescription_c(),
                  weights0->getName().c_str(),
                  weights0Shared,
                  weights->getName().c_str(),
                  weightsShared);
         }
         status = PV_FAILURE;
      }
      if (weights->getNumArbors() != weights0->getNumArbors()) {
         if (mCommunicator->globalCommRank() == 0) {
            ErrorLog().printf(
                  "%s: All connections in the normalization group must have the same number of "
                  "arbors (%s has %d; %s has %d).\n",
                  this->getDescription_c(),
                  weights0->getName().c_str(),
                  weights0->getNumArbors(),
                  weights->getName().c_str(),
                  weights->getNumArbors());
         }
         status = PV_FAILURE;
      }
      if (weights->getNumDataPatches() != weights0->getNumDataPatches()) {
         if (mCommunicator->globalCommRank() == 0) {
            ErrorLog().printf(
                  "%s: All connections in the normalization group must have the same number of "
                  "data patches (%s has %d; %s has %d).\n",
                  this->getDescription_c(),
                  weights0->getName().c_str(),
                  weights0->getNumDataPatches(),
                  weights->getName().c_str(),
                  weights->getNumDataPatches());
         }
         status = PV_FAILURE;
      }
      if (status == PV_FAILURE) {
         MPI_Barrier(mCommunicator->communicator());
         exit(EXIT_FAILURE);
      }
   }

   // Apply rMinX and rMinY
   if (mRMinX > 0.5f && mRMinY > 0.5f) {
      for (auto &weights : mWeightsList) {
         int num_arbors           = weights->getNumArbors();
         int num_patches          = weights->getNumDataPatches();
         int num_weights_in_patch = weights->getPatchSizeOverall();
         for (int arbor = 0; arbor < num_arbors; arbor++) {
            float *dataPatchStart = weights->getData(arbor);
            for (int patchindex = 0; patchindex < num_patches; patchindex++) {
               applyRMin(
                     dataPatchStart + patchindex * num_weights_in_patch,
                     mRMinX,
                     mRMinY,
                     weights->getPatchSizeX(),
                     weights->getPatchSizeY(),
                     weights->getPatchStrideX(),
                     weights->getPatchStrideY());
            }
         }
      }
   }

   // Apply nonnegativeConstraintFlag
   if (mNonnegativeConstraintFlag) {
      for (auto &weights : mWeightsList) {
         int num_arbors           = weights->getNumArbors();
         int num_patches          = weights->getNumDataPatches();
         int num_weights_in_patch = weights->getPatchSizeOverall();
         int num_weights_in_arbor = num_patches * num_weights_in_patch;
         for (int arbor = 0; arbor < num_arbors; arbor++) {
            float *dataStart = weights->getData(arbor);
            for (int weightindex = 0; weightindex < num_weights_in_arbor; weightindex++) {
               float *w = &dataStart[weightindex];
               if (*w < 0) {
                  *w = 0;
               }
            }
         }
      }
   }

   // Apply normalize_cutoff
   if (mNormalizeCutoff > 0) {
      float max = 0.0f;
      for (auto &weights : mWeightsList) {
         int num_arbors           = weights->getNumArbors();
         int num_patches          = weights->getNumDataPatches();
         int num_weights_in_patch = weights->getPatchSizeOverall();
         for (int arbor = 0; arbor < num_arbors; arbor++) {
            float *dataStart = weights->getData(arbor);
            for (int patchindex = 0; patchindex < num_patches; patchindex++) {
               accumulateMaxAbs(
                     dataStart + patchindex * num_weights_in_patch, num_weights_in_patch, &max);
            }
         }
      }
      for (auto &weights : mWeightsList) {
         int num_arbors           = weights->getNumArbors();
         int num_patches          = weights->getNumDataPatches();
         int num_weights_in_patch = weights->getPatchSizeOverall();
         for (int arbor = 0; arbor < num_arbors; arbor++) {
            float *dataStart = weights->getData(arbor);
            for (int patchindex = 0; patchindex < num_patches; patchindex++) {
               applyThreshold(
                     dataStart + patchindex * num_weights_in_patch, num_weights_in_patch, max);
            }
         }
      }
   }

   return PV_SUCCESS;
}

/**
 * Sets all weights in a patch whose absolute value is below a certain value, to zero
 * dataPatchStart is a pointer to a buffer of weights
 * weights_in_patch is the number of weights in the dataPatchStart buffer
 * wMax defines the threshold.  If |w| < wMax * normalize_cutoff, the weight will be zeroed.
 */
int NormalizeMultiply::applyThreshold(float *dataPatchStart, int weights_in_patch, float wMax) {
   assert(mNormalizeCutoff > 0); // Don't call this routine unless normalize_cutoff was set
   float threshold = wMax * mNormalizeCutoff;
   for (int k = 0; k < weights_in_patch; k++) {
      if (fabsf(dataPatchStart[k]) < threshold)
         dataPatchStart[k] = 0;
   }
   return PV_SUCCESS;
}

// dataPatchStart points to head of full-sized patch
// rMinX, rMinY are the minimum radii from the center of the patch,
// all weights inside (non-inclusive) of this radius are set to zero
// the diameter of the central exclusion region is truncated to the nearest integer value, which may
// be zero
int NormalizeMultiply::applyRMin(
      float *dataPatchStart,
      float rMinX,
      float rMinY,
      int nxp,
      int nyp,
      int xPatchStride,
      int yPatchStride) {
   if (rMinX == 0 && rMinY == 0)
      return PV_SUCCESS;
   int fullWidthX        = floor(2 * rMinX);
   int fullWidthY        = floor(2 * rMinY);
   int offsetX           = ceil((nxp - fullWidthX) / 2.0);
   int offsetY           = ceil((nyp - fullWidthY) / 2.0);
   int widthX            = nxp - 2 * offsetX;
   int widthY            = nyp - 2 * offsetY;
   float *rMinPatchStart = dataPatchStart + offsetY * yPatchStride + offsetX * xPatchStride;
   int weights_in_row    = xPatchStride * widthX;
   for (int ky = 0; ky < widthY; ky++) {
      for (int k = 0; k < weights_in_row; k++) {
         rMinPatchStart[k] = 0;
      }
      rMinPatchStart += yPatchStride;
   }
   return PV_SUCCESS;
}

void NormalizeMultiply::normalizePatch(float *patchData, int weightsPerPatch, float multiplier) {
   for (int k = 0; k < weightsPerPatch; k++)
      patchData[k] *= multiplier;
}

} /* namespace PV */
