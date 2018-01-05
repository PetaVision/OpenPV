/*
 * HebbianUpdater.cpp
 *
 *  Created on: Nov 29, 2017
 *      Author: Pete Schultz
 */

#include "HebbianUpdater.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "components/WeightsPair.hpp"
#include "utils/MapLookupByType.hpp"
#include "utils/TransposeWeights.hpp"

namespace PV {

HebbianUpdater::HebbianUpdater(char const *name, HyPerCol *hc) { initialize(name, hc); }

int HebbianUpdater::initialize(char const *name, HyPerCol *hc) {
   return BaseWeightUpdater::initialize(name, hc);
}

int HebbianUpdater::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseWeightUpdater::ioParamsFillGroup(ioFlag);
   ioParam_triggerLayerName(ioFlag);
   ioParam_triggerOffset(ioFlag);
   ioParam_weightUpdatePeriod(ioFlag);
   ioParam_initialWeightUpdateTime(ioFlag);
   ioParam_immediateWeightUpdate(ioFlag);
   ioParam_dWMax(ioFlag);
   ioParam_dWMaxDecayInterval(ioFlag);
   ioParam_dWMaxDecayFactor(ioFlag);
   ioParam_normalizeDw(ioFlag);
   ioParam_useMask(ioFlag);
   ioParam_combine_dW_with_W_flag(ioFlag);
   return PV_SUCCESS;
}

void HebbianUpdater::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      parent->parameters()->ioParamString(
            ioFlag, name, "triggerLayerName", &mTriggerLayerName, nullptr, false /*warnIfAbsent*/);
      if (ioFlag == PARAMS_IO_READ) {
         mTriggerFlag = (mTriggerLayerName != nullptr && mTriggerLayerName[0] != '\0');
      }
   }
}

void HebbianUpdater::ioParam_triggerOffset(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
      if (mTriggerFlag) {
         parent->parameters()->ioParamValue(
               ioFlag, name, "triggerOffset", &mTriggerOffset, mTriggerOffset);
         if (mTriggerOffset < 0) {
            Fatal().printf(
                  "%s error in rank %d process: TriggerOffset (%f) must be positive",
                  getDescription_c(),
                  parent->getCommunicator()->globalCommRank(),
                  mTriggerOffset);
         }
      }
   }
}

void HebbianUpdater::ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
      if (!mTriggerLayerName) {
         parent->parameters()->ioParamValueRequired(
               ioFlag, name, "weightUpdatePeriod", &mWeightUpdatePeriod);
      }
      else
         FatalIf(
               parent->parameters()->present(name, "weightUpdatePeriod"),
               "%s sets both triggerLayerName and weightUpdatePeriod; "
               "only one of these can be set.\n",
               getDescription_c());
   }
}

void HebbianUpdater::ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
      if (!mTriggerLayerName) {
         parent->parameters()->ioParamValue(
               ioFlag,
               name,
               "initialWeightUpdateTime",
               &mInitialWeightUpdateTime,
               mInitialWeightUpdateTime,
               true /*warnIfAbsent*/);
      }
   }
   if (ioFlag == PARAMS_IO_READ) {
      mWeightUpdateTime = mInitialWeightUpdateTime;
   }
}

void HebbianUpdater::ioParam_immediateWeightUpdate(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      parent->parameters()->ioParamValue(
            ioFlag,
            name,
            "immediateWeightUpdate",
            &mImmediateWeightUpdate,
            mImmediateWeightUpdate,
            true /*warnIfAbsent*/);
   }
}

void HebbianUpdater::ioParam_dWMax(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      parent->parameters()->ioParamValueRequired(ioFlag, name, "dWMax", &mDWMax);
   }
}

void HebbianUpdater::ioParam_dWMaxDecayInterval(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "dWMax"));
      if (mDWMax > 0) {
         parent->parameters()->ioParamValue(
               ioFlag,
               name,
               "dWMaxDecayInterval",
               &mDWMaxDecayInterval,
               mDWMaxDecayInterval,
               false);
      }
   }
}

void HebbianUpdater::ioParam_dWMaxDecayFactor(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
      parent->parameters()->ioParamValue(
            ioFlag, name, "dWMaxDecayFactor", &mDWMaxDecayFactor, mDWMaxDecayFactor, false);
      FatalIf(
            mDWMaxDecayFactor < 0.0f || mDWMaxDecayFactor >= 1.0f,
            "%s: dWMaxDecayFactor must be in the interval [0.0, 1.0)\n",
            getName());
   }
}

void HebbianUpdater::ioParam_normalizeDw(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      parent->parameters()->ioParamValue(
            ioFlag, getName(), "normalizeDw", &mNormalizeDw, mNormalizeDw, false /*warnIfAbsent*/);
   }
}

void HebbianUpdater::ioParam_useMask(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
      if (mPlasticityFlag) {
         bool useMask = false;
         parent->parameters()->ioParamValue(
               ioFlag, getName(), "useMask", &useMask, useMask, false /*warnIfAbsent*/);
         if (useMask) {
            if (parent->getCommunicator()->globalCommRank() == 0) {
               ErrorLog().printf("%s has useMask set to true. This parameter is obsolete.\n");
            }
            MPI_Barrier(parent->getCommunicator()->globalCommunicator());
            exit(EXIT_FAILURE);
         }
      }
   }
}

void HebbianUpdater::ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      parent->parameters()->ioParamValue(
            ioFlag,
            name,
            "combine_dW_with_W_flag",
            &mCombine_dWWithWFlag,
            mCombine_dWWithWFlag,
            true /*warnIfAbsent*/);
   }
}

int HebbianUpdater::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status              = PV_SUCCESS;
   auto componentMap       = message->mHierarchy;
   std::string const &desc = getDescription();
   auto *weightsPair       = mapLookupByType<WeightsPair>(componentMap, desc);
   pvAssert(weightsPair);
   if (!weightsPair->getInitInfoCommunicatedFlag()) {
      return PV_POSTPONE;
   }

   status = BaseWeightUpdater::communicateInitInfo(message);
   if (status != PV_SUCCESS) {
      return status;
   }

   weightsPair->needPre();
   mWeights = weightsPair->getPreWeights();
   if (mPlasticityFlag) {
      mWeights->setWeightsArePlastic();
   }
   mWriteCompressedCheckpoints = weightsPair->getWriteCompressedCheckpoints();

   mConnectionData = mapLookupByType<ConnectionData>(message->mHierarchy, getDescription());
   pvAssert(mConnectionData != nullptr);

   if (mTriggerFlag) {
      auto *objectMapComponent = mapLookupByType<ObjectMapComponent>(componentMap, desc);
      pvAssert(objectMapComponent);
      mTriggerLayer = objectMapComponent->lookup<HyPerLayer>(std::string(mTriggerLayerName));

      // Although weightUpdatePeriod and weightUpdateTime are being set here, if triggerLayerName
      // is set, they are not being used. Only updating for backwards compatibility
      mWeightUpdatePeriod = mTriggerLayer->getDeltaUpdateTime();
      if (mWeightUpdatePeriod <= 0) {
         if (mPlasticityFlag == true) {
            WarnLog() << "Connection " << name << "triggered layer " << mTriggerLayerName
                      << " never updates, turning plasticity flag off\n";
            mPlasticityFlag = false;
         }
      }
      if (mWeightUpdatePeriod != -1 && mTriggerOffset >= mWeightUpdatePeriod) {
         Fatal().printf(
               "%s, rank %d process: TriggerOffset (%f) must be lower than the change in update "
               "time (%f) of the attached trigger layer\n",
               getDescription_c(),
               parent->getCommunicator()->globalCommRank(),
               mTriggerOffset,
               mWeightUpdatePeriod);
      }
      mWeightUpdateTime = parent->getDeltaTime();
   }

   return status;
}

void HebbianUpdater::addClone(ConnectionData *connectionData) {

   // CloneConn's communicateInitInfo makes sure the pre layers' borders are in sync,
   // but for PlasticCloneConns to apply the update rules correctly, we need the
   // post layers' borders to be equal as well.

   pvAssert(connectionData->getInitInfoCommunicatedFlag());
   pvAssert(mConnectionData->getInitInfoCommunicatedFlag());
   connectionData->getPost()->synchronizeMarginWidth(mConnectionData->getPost());
   mConnectionData->getPost()->synchronizeMarginWidth(connectionData->getPost());

   // Add the new connection data to the list of clones.
   mClones.push_back(connectionData);
}

int HebbianUpdater::allocateDataStructures() {
   int status = BaseWeightUpdater::allocateDataStructures();
   if (status != PV_SUCCESS) {
      return status;
   }
   if (mPlasticityFlag) {
      if (mCombine_dWWithWFlag) {
         mDeltaWeights = mWeights;
      }
      else {
         if (mWeights->getGeometry() == nullptr) {
            return PV_POSTPONE;
         }
         mDeltaWeights = new Weights(name);
         mDeltaWeights->initialize(mWeights);
         mDeltaWeights->setMargins(
               mConnectionData->getPre()->getLayerLoc()->halo,
               mConnectionData->getPost()->getLayerLoc()->halo);
         mDeltaWeights->allocateDataStructures();
      }
      if (mWeights->getSharedFlag() && mNormalizeDw) {
         int const nPatches       = mDeltaWeights->getNumDataPatches();
         int const numArbors      = mConnectionData->getNumAxonalArbors();
         mNumKernelActivations    = (long **)pvCalloc(numArbors, sizeof(long *));
         int const sp             = mDeltaWeights->getPatchSizeOverall();
         std::size_t numWeights   = (std::size_t)(sp) * (std::size_t)nPatches;
         mNumKernelActivations[0] = (long *)pvCalloc(numWeights, sizeof(long));
         for (int arborId = 0; arborId < numArbors; arborId++) {
            mNumKernelActivations[arborId] = (mNumKernelActivations[0] + sp * nPatches * arborId);
         } // loop over arbors
      }
   }

   if (mPlasticityFlag && !mTriggerLayer) {
      if (mWeightUpdateTime < parent->simulationTime()) {
         while (mWeightUpdateTime <= parent->simulationTime()) {
            mWeightUpdateTime += mWeightUpdatePeriod;
         }
         if (parent->getCommunicator()->globalCommRank() == 0) {
            WarnLog().printf(
                  "initialWeightUpdateTime of %s less than simulation start time.  Adjusting "
                  "weightUpdateTime to %f\n",
                  getDescription_c(),
                  mWeightUpdateTime);
         }
      }
      mLastUpdateTime = mWeightUpdateTime - parent->getDeltaTime();
   }
   mLastTimeUpdateCalled = parent->simulationTime();

   return status;
}

int HebbianUpdater::registerData(Checkpointer *checkpointer) {
   int status = BaseWeightUpdater::registerData(checkpointer);
   if (status != PV_SUCCESS) {
      return status;
   }
   if (mPlasticityFlag and !mImmediateWeightUpdate) {
      mDeltaWeights->checkpointWeightPvp(checkpointer, "dW", mWriteCompressedCheckpoints);
      // Do we need to get PrepareCheckpointWrite messages, to call blockingNormalize_dW()?
   }
   std::string nameString = std::string(name);
   checkpointer->registerCheckpointData(
         nameString,
         "lastUpdateTime",
         &mLastUpdateTime,
         (std::size_t)1,
         true /*broadcast*/,
         false /*not constant*/);
   if (mPlasticityFlag && !mTriggerLayer) {
      checkpointer->registerCheckpointData(
            nameString,
            "weightUpdateTime",
            &mWeightUpdateTime,
            (std::size_t)1,
            true /*broadcast*/,
            false /*not constant*/);
   }
   return status;
}

int HebbianUpdater::readStateFromCheckpoint(Checkpointer *checkpointer) {
   if (mConnectionData->getInitializeFromCheckpointFlag()) {
      if (mPlasticityFlag and !mImmediateWeightUpdate) {
         checkpointer->readNamedCheckpointEntry(
               std::string(name), std::string("dW"), false /*not constant*/);
      }
   }
   return PV_SUCCESS;
}

void HebbianUpdater::updateState(double simTime, double dt) {
   if (needUpdate(simTime, dt)) {
      pvAssert(mPlasticityFlag);
      if (mImmediateWeightUpdate) {
         updateWeightsImmediate(simTime, dt);
      }
      else {
         updateWeightsDelayed(simTime, dt);
      }

      decay_dWMax();

      mLastUpdateTime = simTime;
      mWeights->setTimestamp(simTime);
      computeNewWeightUpdateTime(simTime, mWeightUpdateTime);
      mNeedFinalize = true;
   }
   mLastTimeUpdateCalled = simTime;
}

bool HebbianUpdater::needUpdate(double simTime, double dt) {
   if (!mPlasticityFlag) {
      return false;
   }
   if (mTriggerLayer) {
      return mTriggerLayer->needUpdate(simTime + mTriggerOffset, dt);
   }
   return simTime >= mWeightUpdateTime;
}

void HebbianUpdater::updateWeightsImmediate(double simTime, double dt) {
   updateLocal_dW();
   reduce_dW();
   blockingNormalize_dW();
   updateArbors();
}

void HebbianUpdater::updateWeightsDelayed(double simTime, double dt) {
   blockingNormalize_dW();
   updateArbors();
   updateLocal_dW();
   reduce_dW();
}

void HebbianUpdater::updateLocal_dW() {
   pvAssert(mPlasticityFlag);
   int status          = PV_SUCCESS;
   int const numArbors = mConnectionData->getNumAxonalArbors();
   for (int arborId = 0; arborId < numArbors; arborId++) {
      status = initialize_dW(arborId);
      if (status == PV_BREAK) {
         status = PV_SUCCESS;
         break;
      }
   }
   pvAssert(status == PV_SUCCESS);

   for (int arborId = 0; arborId < numArbors; arborId++) {
      status = update_dW(arborId);
      if (status == PV_BREAK) {
         break;
      }
   }
   pvAssert(status == PV_SUCCESS or status == PV_BREAK);
}

int HebbianUpdater::initialize_dW(int arborId) {
   if (!mCombine_dWWithWFlag) {
      clear_dW(arborId);
   }
   if (mNumKernelActivations) {
      clearNumActivations(arborId);
   }
   // default initialize_dW returns PV_BREAK
   return PV_BREAK;
}

int HebbianUpdater::clear_dW(int arborId) {
   // zero out all dW.
   // This also zeroes out the unused parts of shrunken patches
   int const syPatch        = mWeights->getPatchStrideY();
   int const nxp            = mWeights->getPatchSizeX();
   int const nyp            = mWeights->getPatchSizeY();
   int const nfp            = mWeights->getPatchSizeF();
   int const nkPatch        = nfp * nxp;
   int const numArbors      = mConnectionData->getNumAxonalArbors();
   int const numDataPatches = mWeights->getNumDataPatches();
   for (int kArbor = 0; kArbor < numArbors; kArbor++) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int kKernel = 0; kKernel < numDataPatches; kKernel++) {
         float *dWeights = mDeltaWeights->getDataFromDataIndex(kArbor, kKernel);
         for (int kyPatch = 0; kyPatch < nyp; kyPatch++) {
            for (int kPatch = 0; kPatch < nkPatch; kPatch++) {
               dWeights[kyPatch * syPatch + kPatch] = 0.0f;
            }
         }
      }
   }
   return PV_BREAK;
}

int HebbianUpdater::clearNumActivations(int arborId) {
   // zero out all dW.
   // This also zeroes out the unused parts of shrunken patches
   int const syPatch          = mWeights->getPatchStrideY();
   int const nxp              = mWeights->getPatchSizeX();
   int const nyp              = mWeights->getPatchSizeY();
   int const nfp              = mWeights->getPatchSizeF();
   int const nkPatch          = nfp * nxp;
   int const patchSizeOverall = nyp * nkPatch;
   int const numArbors        = mConnectionData->getNumAxonalArbors();
   int const numDataPatches   = mWeights->getNumDataPatches();
   for (int kArbor = 0; kArbor < numArbors; kArbor++) {
      for (int kKernel = 0; kKernel < numDataPatches; kKernel++) {
         long *activations = &mNumKernelActivations[kArbor][kKernel * patchSizeOverall];
         // long *activations = getActivationsHead(kArbor, kKernel);
         for (int kyPatch = 0; kyPatch < nyp; kyPatch++) {
            for (int kPatch = 0; kPatch < nkPatch; kPatch++) {
               activations[kPatch] = 0.0f;
            }
            activations += syPatch;
         }
      }
   }
   return PV_BREAK;
}

int HebbianUpdater::update_dW(int arborID) {
   // compute dW but don't add them to the weights yet.
   // That takes place in reduceKernels, so that the output is
   // independent of the number of processors.
   HyPerLayer *pre       = mConnectionData->getPre();
   HyPerLayer *post      = mConnectionData->getPost();
   int nExt              = pre->getNumExtended();
   PVLayerLoc const *loc = pre->getLayerLoc();
   int const nbatch      = loc->nbatch;
   int delay             = mConnectionData->getDelay(arborID);

   float const *preactbufHead  = pre->getLayerData(delay);
   float const *postactbufHead = post->getLayerData();

   if (mWeights->getSharedFlag()) {
      // Calculate x and y cell size
      int xCellSize  = zUnitCellSize(pre->getXScale(), post->getXScale());
      int yCellSize  = zUnitCellSize(pre->getYScale(), post->getYScale());
      int nxExt      = loc->nx + loc->halo.lt + loc->halo.rt;
      int nyExt      = loc->ny + loc->halo.up + loc->halo.dn;
      int nf         = loc->nf;
      int numKernels = mWeights->getNumDataPatches();

      for (int b = 0; b < nbatch; b++) {
// Shared weights done in parallel, parallel in numkernels
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int kernelIdx = 0; kernelIdx < numKernels; kernelIdx++) {

            // Calculate xCellIdx, yCellIdx, and fCellIdx from kernelIndex
            int kxCellIdx = kxPos(kernelIdx, xCellSize, yCellSize, nf);
            int kyCellIdx = kyPos(kernelIdx, xCellSize, yCellSize, nf);
            int kfIdx     = featureIndex(kernelIdx, xCellSize, yCellSize, nf);
            // Loop over all cells in pre ext
            int kyIdx    = kyCellIdx;
            int yCellIdx = 0;
            while (kyIdx < nyExt) {
               int kxIdx    = kxCellIdx;
               int xCellIdx = 0;
               while (kxIdx < nxExt) {
                  // Calculate kExt from ky, kx, and kf
                  int kExt = kIndex(kxIdx, kyIdx, kfIdx, nxExt, nyExt, nf);
                  updateInd_dW(arborID, b, preactbufHead, postactbufHead, kExt);
                  xCellIdx++;
                  kxIdx = kxCellIdx + xCellIdx * xCellSize;
               }
               yCellIdx++;
               kyIdx = kyCellIdx + yCellIdx * yCellSize;
            }
         }
      }
   }
   else {
      for (int b = 0; b < nbatch; b++) {
// Shared weights done in parallel, parallel in numkernels
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int kExt = 0; kExt < nExt; kExt++) {
            updateInd_dW(arborID, b, preactbufHead, postactbufHead, kExt);
         }
      }
   }

   // If update from clones, update dw here as well
   // Updates on all PlasticClones
   for (auto &c : mClones) {
      pvAssert(c->getPre()->getNumExtended() == nExt);
      pvAssert(c->getPre()->getLayerLoc()->nbatch == nbatch);
      float const *clonePre  = c->getPre()->getLayerData(delay);
      float const *clonePost = c->getPost()->getLayerData();
      for (int b = 0; b < nbatch; b++) {
         for (int kExt = 0; kExt < nExt; kExt++) {
            updateInd_dW(arborID, b, clonePre, clonePost, kExt);
         }
      }
   }

   return PV_SUCCESS;
}

int HebbianUpdater::updateInd_dW(
      int arborID,
      int batchID,
      float const *preLayerData,
      float const *postLayerData,
      int kExt) {
   HyPerLayer *pre           = mConnectionData->getPre();
   HyPerLayer *post          = mConnectionData->getPost();
   const PVLayerLoc *postLoc = post->getLayerLoc();

   const float *maskactbuf = NULL;
   const float *preactbuf  = preLayerData + batchID * pre->getNumExtended();
   const float *postactbuf = postLayerData + batchID * post->getNumExtended();

   int sya = (postLoc->nf * (postLoc->nx + postLoc->halo.lt + postLoc->halo.rt));

   float preact = preactbuf[kExt];
   if (preact == 0.0f) {
      return PV_CONTINUE;
   }

   Patch const &patch = mWeights->getPatch(kExt);
   int ny             = patch.ny;
   int nk             = patch.nx * mWeights->getPatchSizeF();
   if (ny == 0 || nk == 0) {
      return PV_SUCCESS;
   }

   size_t offset           = mWeights->getGeometry()->getAPostOffset(kExt);
   const float *postactRef = &postactbuf[offset];

   int sym                 = 0;
   const float *maskactRef = NULL;

   float *dwdata =
         mDeltaWeights->getDataFromPatchIndex(arborID, kExt) + mDeltaWeights->getPatch(kExt).offset;
   long *activations = nullptr;
   if (mWeights->getSharedFlag() && mNormalizeDw) {
      int dataIndex        = mWeights->calcDataIndexFromPatchIndex(kExt);
      int patchSizeOverall = mWeights->getPatchSizeOverall();
      int patchOffset      = mWeights->getPatch(kExt).offset;
      activations = &mNumKernelActivations[arborID][dataIndex * patchSizeOverall + patchOffset];
   }

   int syp         = mWeights->getPatchStrideY();
   int lineoffsetw = 0;
   int lineoffseta = 0;
   int lineoffsetm = 0;
   for (int y = 0; y < ny; y++) {
      for (int k = 0; k < nk; k++) {
         float aPost = postactRef[lineoffseta + k];
         // calculate contribution to dw
         // Note: this is a hack, as batching calls this function, but overwrites to allocate
         // numKernelActivations with non-shared weights
         if (activations) {
            // Offset in the case of a shrunken patch, where dwdata is applying when calling
            // getDeltaWeightsData
            activations[lineoffsetw + k]++;
         }
         dwdata[lineoffsetw + k] += updateRule_dW(preact, aPost);
      }
      lineoffsetw += syp;
      lineoffseta += sya;
      lineoffsetm += sym;
   }
   return PV_SUCCESS;
}

float HebbianUpdater::updateRule_dW(float pre, float post) { return mDWMax * pre * post; }

void HebbianUpdater::reduce_dW() {
   int status          = PV_SUCCESS;
   int const numArbors = mConnectionData->getNumAxonalArbors();
   for (int arborId = 0; arborId < numArbors; arborId++) {
      status = reduce_dW(arborId);
      if (status == PV_BREAK) {
         break;
      }
   }
   pvAssert(status == PV_SUCCESS or status == PV_BREAK);
   mReductionPending = true;
}

int HebbianUpdater::reduce_dW(int arborId) {
   int kernel_status = PV_BREAK;
   if (mWeights->getSharedFlag()) {
      kernel_status = reduceKernels(arborId); // combine partial changes in each column
      if (mNormalizeDw) {
         int activation_status = reduceActivations(arborId);
         pvAssert(kernel_status == activation_status);
      }
   }
   else {
      reduceAcrossBatch(arborId);
   }
   return kernel_status;
}

int HebbianUpdater::reduceKernels(int arborID) {
   pvAssert(mWeights->getSharedFlag() && mPlasticityFlag);
   Communicator *comm = parent->getCommunicator();
   const int nxProcs  = comm->numCommColumns();
   const int nyProcs  = comm->numCommRows();
   const int nbProcs  = comm->numCommBatches();
   const int nProcs   = nxProcs * nyProcs * nbProcs;
   if (nProcs != 1) {
      const MPI_Comm mpi_comm = comm->globalCommunicator();
      const int numPatches    = mWeights->getNumDataPatches();
      const size_t patchSize  = (size_t)mWeights->getPatchSizeOverall();
      const size_t localSize  = (size_t)numPatches * (size_t)patchSize;
      const size_t arborSize  = localSize * (size_t)mConnectionData->getNumAxonalArbors();

      auto sz = mDeltaWeightsReduceRequests.size();
      mDeltaWeightsReduceRequests.resize(sz + 1);
      MPI_Iallreduce(
            MPI_IN_PLACE,
            mDeltaWeights->getData(arborID),
            arborSize,
            MPI_FLOAT,
            MPI_SUM,
            mpi_comm,
            &(mDeltaWeightsReduceRequests.data())[sz]);
   }

   return PV_BREAK;
}

int HebbianUpdater::reduceActivations(int arborID) {
   pvAssert(mWeights->getSharedFlag() && mPlasticityFlag);
   Communicator *comm = parent->getCommunicator();
   const int nxProcs  = comm->numCommColumns();
   const int nyProcs  = comm->numCommRows();
   const int nbProcs  = comm->numCommBatches();
   const int nProcs   = nxProcs * nyProcs * nbProcs;
   if (mNumKernelActivations && nProcs != 1) {
      const MPI_Comm mpi_comm = comm->globalCommunicator();
      const int numPatches    = mWeights->getNumDataPatches();
      const size_t patchSize  = (size_t)mWeights->getPatchSizeOverall();
      const size_t localSize  = numPatches * patchSize;
      const size_t arborSize  = localSize * mConnectionData->getNumAxonalArbors();

      auto sz = mDeltaWeightsReduceRequests.size();
      mDeltaWeightsReduceRequests.resize(sz + 1);
      MPI_Iallreduce(
            MPI_IN_PLACE,
            mNumKernelActivations[arborID],
            arborSize,
            MPI_LONG,
            MPI_SUM,
            mpi_comm,
            &(mDeltaWeightsReduceRequests.data())[sz]);
   }

   return PV_BREAK;
}

void HebbianUpdater::reduceAcrossBatch(int arborID) {
   pvAssert(!mWeights->getSharedFlag() && mPlasticityFlag);
   if (parent->getCommunicator()->numCommBatches() != 1) {
      const int numPatches     = mWeights->getNumDataPatches();
      const size_t patchSize   = (size_t)mWeights->getPatchSizeOverall();
      size_t const localSize   = (size_t)numPatches * (size_t)patchSize;
      size_t const arborSize   = localSize * (size_t)mConnectionData->getNumAxonalArbors();
      MPI_Comm const batchComm = parent->getCommunicator()->batchCommunicator();

      auto sz = mDeltaWeightsReduceRequests.size();
      mDeltaWeightsReduceRequests.resize(sz + 1);
      MPI_Iallreduce(
            MPI_IN_PLACE,
            mDeltaWeights->getData(arborID),
            arborSize,
            MPI_FLOAT,
            MPI_SUM,
            batchComm,
            &(mDeltaWeightsReduceRequests.data())[sz]);
   }
}

void HebbianUpdater::blockingNormalize_dW() {
   if (mReductionPending) {
      wait_dWReduceRequests();
      normalize_dW();
      mReductionPending = false;
   }
}

void HebbianUpdater::wait_dWReduceRequests() {
   MPI_Waitall(
         mDeltaWeightsReduceRequests.size(),
         mDeltaWeightsReduceRequests.data(),
         MPI_STATUSES_IGNORE);
   mDeltaWeightsReduceRequests.clear();
}

void HebbianUpdater::normalize_dW() {
   int status = PV_SUCCESS;
   if (mNormalizeDw) {
      int const numArbors = mConnectionData->getNumAxonalArbors();
      for (int arborId = 0; arborId < numArbors; arborId++) {
         status = normalize_dW(arborId);
         if (status == PV_BREAK) {
            break;
         }
      }
   }
   pvAssert(status == PV_SUCCESS or status == PV_BREAK);
}

int HebbianUpdater::normalize_dW(int arbor_ID) {
   // This is here in case other classes overwrite the outer class calling this function
   if (!mNormalizeDw) {
      return PV_SUCCESS;
   }
   if (mWeights->getSharedFlag()) {
      pvAssert(mNumKernelActivations);
      int numKernelIndices = mWeights->getNumDataPatches();
      int const numArbors  = mConnectionData->getNumAxonalArbors();
      for (int loop_arbor = 0; loop_arbor < numArbors; loop_arbor++) {
// Divide by numKernelActivations in this timestep
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int kernelindex = 0; kernelindex < numKernelIndices; kernelindex++) {
            // Calculate pre feature index from patch index
            int numpatchitems  = mWeights->getPatchSizeOverall();
            float *dwpatchdata = mDeltaWeights->getDataFromDataIndex(loop_arbor, kernelindex);
            long *activations  = &mNumKernelActivations[loop_arbor][kernelindex * numpatchitems];
            for (int n = 0; n < numpatchitems; n++) {
               long divisor = activations[n];

               if (divisor != 0) {
                  dwpatchdata[n] /= divisor;
               }
               else {
                  dwpatchdata[n] = 0;
               }
            }
         }
      }
   }
   // TODO: non-shared weights should divide by batch period if applicable
   return PV_BREAK;
}

void HebbianUpdater::updateArbors() {
   int status          = PV_SUCCESS;
   int const numArbors = mConnectionData->getNumAxonalArbors();
   for (int arborId = 0; arborId < numArbors; arborId++) {
      status = updateWeights(arborId); // Apply changes in weights
      if (status == PV_BREAK) {
         status = PV_SUCCESS;
         break;
      }
   }
   pvAssert(status == PV_SUCCESS or status == PV_BREAK);
}

int HebbianUpdater::updateWeights(int arborId) {
   // add dw to w
   int const numArbors       = mConnectionData->getNumAxonalArbors();
   int const weightsPerArbor = mWeights->getNumDataPatches() * mWeights->getPatchSizeOverall();
   for (int kArbor = 0; kArbor < numArbors; kArbor++) {
      float *w_data_start = mWeights->getData(kArbor);
      for (long int k = 0; k < weightsPerArbor; k++) {
         w_data_start[k] += mDeltaWeights->getData(kArbor)[k];
      }
   }
   return PV_BREAK;
}

void HebbianUpdater::decay_dWMax() {
   if (mDWMaxDecayInterval > 0) {
      if (--mDWMaxDecayTimer < 0) {
         float oldDWMax   = mDWMax;
         mDWMaxDecayTimer = mDWMaxDecayInterval;
         mDWMax *= 1.0f - mDWMaxDecayFactor;
         InfoLog() << getName() << ": dWMax decayed from " << oldDWMax << " to " << mDWMax << "\n";
      }
   }
}

void HebbianUpdater::computeNewWeightUpdateTime(double simTime, double currentUpdateTime) {
   // Only called if plasticity flag is set
   if (!mTriggerLayer) {
      while (simTime >= mWeightUpdateTime) {
         mWeightUpdateTime += mWeightUpdatePeriod;
      }
   }
}

} // namespace PV
