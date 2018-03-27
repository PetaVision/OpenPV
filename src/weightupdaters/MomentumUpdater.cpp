/*
 * MomentumUpdater.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "MomentumUpdater.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

MomentumUpdater::MomentumUpdater(char const *name, HyPerCol *hc) { initialize(name, hc); }

int MomentumUpdater::initialize(char const *name, HyPerCol *hc) {
   return HebbianUpdater::initialize(name, hc);
}

void MomentumUpdater::setObjectType() { mObjectType = "MomentumUpdater"; }

int MomentumUpdater::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HebbianUpdater::ioParamsFillGroup(ioFlag);
   ioParam_momentumMethod(ioFlag);
   ioParam_momentumTau(ioFlag);
   ioParam_momentumDecay(ioFlag);
   return status;
}

void MomentumUpdater::ioParam_momentumMethod(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      parent->parameters()->ioParamStringRequired(ioFlag, name, "momentumMethod", &mMomentumMethod);
      if (strcmp(mMomentumMethod, "simple") == 0) {
         mMethod = SIMPLE;
      }
      else if (strcmp(mMomentumMethod, "viscosity") == 0) {
         mMethod = VISCOSITY;
      }
      else if (strcmp(mMomentumMethod, "alex") == 0) {
         mMethod = ALEX;
      }
      else {
         Fatal() << "MomentumUpdater " << name << ": momentumMethod of " << mMomentumMethod
                 << " is not known. Options are \"simple\", \"viscosity\", and \"alex\".\n";
      }
   }
}

void MomentumUpdater::ioParam_momentumTau(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "momentumMethod"));
      float defaultVal = 0;
      switch (mMethod) {
         case SIMPLE: defaultVal    = 0.25f; break;
         case VISCOSITY: defaultVal = 100.0f; break;
         case ALEX: defaultVal      = 0.9f; break;
         default: pvAssertMessage(0, "Unrecognized momentumMethod\n"); break;
      }

      parent->parameters()->ioParamValue(ioFlag, name, "momentumTau", &mMomentumTau, defaultVal);
   }
}

void MomentumUpdater::ioParam_momentumDecay(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      parent->parameters()->ioParamValue(
            ioFlag, name, "momentumDecay", &mMomentumDecay, mMomentumDecay);
      if (mMomentumDecay < 0.0f || mMomentumDecay > 1.0f) {
         Fatal() << "MomentumUpdater " << name
                 << ": momentumDecay must be between 0 and 1 inclusive\n";
      }
   }
}

Response::Status MomentumUpdater::allocateDataStructures() {
   auto status = HebbianUpdater::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   if (!mPlasticityFlag) {
      return status;
   }
   mPrevDeltaWeights = new Weights(name);
   mPrevDeltaWeights->initialize(mWeights);
   mPrevDeltaWeights->setMargins(
         mConnectionData->getPre()->getLayerLoc()->halo,
         mConnectionData->getPost()->getLayerLoc()->halo);
   mPrevDeltaWeights->allocateDataStructures();
   return Response::SUCCESS;
}

Response::Status MomentumUpdater::registerData(Checkpointer *checkpointer) {
   auto status = HebbianUpdater::registerData(checkpointer);
   if (!Response::completed(status)) {
      return status;
   }
   // Note: HebbianUpdater does not checkpoint dW if the mImmediateWeightUpdate flag is true.
   // Do we need to handle it here and in readStateFromCheckpoint? --pschultz, 2017-12-16
   if (mPlasticityFlag) {
      mPrevDeltaWeights->checkpointWeightPvp(checkpointer, "prev_dW", mWriteCompressedCheckpoints);
   }
   return Response::SUCCESS;
}

Response::Status MomentumUpdater::readStateFromCheckpoint(Checkpointer *checkpointer) {
   if (mInitializeFromCheckpointFlag) {
      // Note: HebbianUpdater does not checkpoint dW if the mImmediateWeightUpdate flag is true.
      // Do we need to handle it here and in registerData? --pschultz, 2017-12-16
      if (mPlasticityFlag) {
         checkpointer->readNamedCheckpointEntry(
               std::string(name), std::string("prev_dW"), false /*not constant*/);
      }
      return Response::SUCCESS;
   }
   else {
      return Response::NO_ACTION;
   }
}

int MomentumUpdater::updateWeights(int arborId) {
   // Add momentum right before updateWeights
   applyMomentum(arborId);

   // Current dW saved to prev_dW
   pvAssert(mPrevDeltaWeights);
   std::memcpy(
         mPrevDeltaWeights->getData(arborId),
         mDeltaWeights->getDataReadOnly(arborId),
         sizeof(float) * mDeltaWeights->getPatchSizeOverall() * mDeltaWeights->getNumDataPatches());

   // add dw to w
   return HebbianUpdater::updateWeights(arborId);
}

void MomentumUpdater::applyMomentum(int arborId) {
   // Shared weights done in parallel, parallel in numkernels
   switch (mMethod) {
      case SIMPLE: applyMomentum(arborId, mMomentumTau, mMomentumDecay); break;
      case VISCOSITY: applyMomentum(arborId, std::exp(-1.0f / mMomentumTau), mMomentumDecay); break;
      case ALEX: applyMomentum(arborId, mMomentumTau, mMomentumDecay * mDWMax); break;
      default: pvAssertMessage(0, "Unrecognized momentumMethod\n"); break;
   }
}

void MomentumUpdater::applyMomentum(int arborId, float dwFactor, float wFactor) {
   int const numKernels = mDeltaWeights->getNumDataPatches();
   pvAssert(numKernels == mPrevDeltaWeights->getNumDataPatches());
   int const patchSizeOverall = mDeltaWeights->getPatchSizeOverall();
   pvAssert(patchSizeOverall == mPrevDeltaWeights->getPatchSizeOverall());
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for (int kernelIdx = 0; kernelIdx < numKernels; kernelIdx++) {
      float *dwdata_start        = mDeltaWeights->getDataFromDataIndex(arborId, kernelIdx);
      float const *prev_dw_start = mPrevDeltaWeights->getDataFromDataIndex(arborId, kernelIdx);
      float const *wdata_start   = mWeights->getDataFromDataIndex(arborId, kernelIdx);
      for (int k = 0; k < patchSizeOverall; k++) {
         dwdata_start[k] += dwFactor * prev_dw_start[k] - wFactor * wdata_start[k];
      }
   }
   // Since weights data is allocated with all patches of a given arbor in a single vector,
   // these two for-loops can probably be collapsed. --pschultz 2017-12-16
}

} // namespace PV
