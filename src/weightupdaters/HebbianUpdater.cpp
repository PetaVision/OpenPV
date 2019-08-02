/*
 * HebbianUpdater.cpp
 *
 *  Created on: Nov 29, 2017
 *      Author: Pete Schultz
 */

#include "HebbianUpdater.hpp"
#include "components/WeightsPair.hpp"
#include "observerpattern/ObserverTable.hpp"
#include "utils/TransposeWeights.hpp"

namespace PV {

HebbianUpdater::HebbianUpdater(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

HebbianUpdater::~HebbianUpdater() { cleanup(); }

void HebbianUpdater::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseWeightUpdater::initialize(name, params, comm);
}

void HebbianUpdater::setObjectType() { mObjectType = "HebbianUpdater"; }

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
   return status;
}

void HebbianUpdater::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      parameters()->ioParamString(
            ioFlag, name, "triggerLayerName", &mTriggerLayerName, nullptr, false /*warnIfAbsent*/);
      if (ioFlag == PARAMS_IO_READ) {
         mTriggerFlag = (mTriggerLayerName != nullptr && mTriggerLayerName[0] != '\0');
      }
   }
}

void HebbianUpdater::ioParam_triggerOffset(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      pvAssert(!parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
      if (mTriggerFlag) {
         parameters()->ioParamValue(ioFlag, name, "triggerOffset", &mTriggerOffset, mTriggerOffset);
         if (mTriggerOffset < 0) {
            Fatal().printf(
                  "%s error in rank %d process: TriggerOffset (%f) must be positive",
                  getDescription_c(),
                  mCommunicator->globalCommRank(),
                  mTriggerOffset);
         }
      }
   }
}

void HebbianUpdater::ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      pvAssert(!parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
      if (!mTriggerLayerName) {
         parameters()->ioParamValueRequired(
               ioFlag, name, "weightUpdatePeriod", &mWeightUpdatePeriod);
      }
      else
         FatalIf(
               parameters()->present(name, "weightUpdatePeriod"),
               "%s sets both triggerLayerName and weightUpdatePeriod; "
               "only one of these can be set.\n",
               getDescription_c());
   }
}

void HebbianUpdater::ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      pvAssert(!parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
      if (!mTriggerLayerName) {
         parameters()->ioParamValue(
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
   pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      parameters()->ioParamValue(
            ioFlag,
            name,
            "immediateWeightUpdate",
            &mImmediateWeightUpdate,
            mImmediateWeightUpdate,
            true /*warnIfAbsent*/);
   }
}

void HebbianUpdater::ioParam_dWMax(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      parameters()->ioParamValueRequired(ioFlag, name, "dWMax", &mDWMax);
   }
}

void HebbianUpdater::ioParam_dWMaxDecayInterval(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      pvAssert(!parameters()->presentAndNotBeenRead(name, "dWMax"));
      if (mDWMax > 0) {
         parameters()->ioParamValue(
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
   pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
      parameters()->ioParamValue(
            ioFlag, name, "dWMaxDecayFactor", &mDWMaxDecayFactor, mDWMaxDecayFactor, false);
      FatalIf(
            mDWMaxDecayFactor < 0.0f || mDWMaxDecayFactor >= 1.0f,
            "%s: dWMaxDecayFactor must be in the interval [0.0, 1.0)\n",
            getName());
   }
}

void HebbianUpdater::ioParam_normalizeDw(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      parameters()->ioParamValue(
            ioFlag, getName(), "normalizeDw", &mNormalizeDw, mNormalizeDw, false /*warnIfAbsent*/);
   }
}

void HebbianUpdater::ioParam_useMask(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
      if (mPlasticityFlag) {
         bool useMask = false;
         parameters()->ioParamValue(
               ioFlag, getName(), "useMask", &useMask, useMask, false /*warnIfAbsent*/);
         if (useMask) {
            if (mCommunicator->globalCommRank() == 0) {
               ErrorLog().printf("%s has useMask set to true. This parameter is obsolete.\n");
            }
            MPI_Barrier(mCommunicator->globalCommunicator());
            exit(EXIT_FAILURE);
         }
      }
   }
}

void HebbianUpdater::ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      parameters()->ioParamValue(
            ioFlag,
            name,
            "combine_dW_with_W_flag",
            &mCombine_dWWithWFlag,
            mCombine_dWWithWFlag,
            true /*warnIfAbsent*/);
   }
}

Response::Status
HebbianUpdater::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *objectTable = message->mObjectTable;
   auto *weightsPair = objectTable->findObject<WeightsPair>(getName());
   pvAssert(weightsPair);
   if (!weightsPair->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   pvAssert(mInitializeFromCheckpointFlag == weightsPair->getInitializeFromCheckpointFlag());

   auto status = BaseWeightUpdater::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   weightsPair->needPre();
   mWeights = weightsPair->getPreWeights();
   if (mPlasticityFlag) {
      mWeights->setWeightsArePlastic();
   }
   mWriteCompressedCheckpoints = weightsPair->getWriteCompressedCheckpoints();

   mConnectionData = objectTable->findObject<ConnectionData>(getName());
   FatalIf(
         mConnectionData == nullptr,
         "%s requires a ConnectionData component.\n",
         getDescription_c());

   mArborList = message->mObjectTable->findObject<ArborList>(getName());
   FatalIf(mArborList == nullptr, "%s requires a ArborList component.\n", getDescription_c());

   if (mTriggerFlag) {
      mTriggerControl = objectTable->findObject<LayerUpdateController>(mTriggerLayerName);
      FatalIf(
            mTriggerControl == nullptr,
            "%s: triggerLayerName \"%s\" does not have a LayerUpdateController.\n",
            getDescription_c(),
            mTriggerLayerName);
   }

   return Response::SUCCESS;
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

Response::Status HebbianUpdater::allocateDataStructures() {
   auto status = BaseWeightUpdater::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   if (mPlasticityFlag) {
      if (mCombine_dWWithWFlag) {
         mDeltaWeights = mWeights;
      }
      else {
         if (mWeights->getGeometry() == nullptr) {
            return status + Response::POSTPONE;
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
         int const numArbors      = mArborList->getNumAxonalArbors();
         mNumKernelActivations    = (long **)pvCalloc(numArbors, sizeof(long *));
         int const sp             = mDeltaWeights->getPatchSizeOverall();
         std::size_t numWeights   = (std::size_t)(sp) * (std::size_t)nPatches;
         mNumKernelActivations[0] = (long *)pvCalloc(numWeights, sizeof(long));
         for (int arborId = 0; arborId < numArbors; arborId++) {
            mNumKernelActivations[arborId] = (mNumKernelActivations[0] + sp * nPatches * arborId);
         } // loop over arbors
      }
   }

   if (mPlasticityFlag && !mTriggerControl) {
      if (mWeightUpdateTime < 0.0) {
         while (mWeightUpdateTime <= 0.0) {
            mWeightUpdateTime += mWeightUpdatePeriod;
         }
         if (mCommunicator->globalCommRank() == 0) {
            WarnLog().printf(
                  "initialWeightUpdateTime of %s less than simulation start time.  Adjusting "
                  "weightUpdateTime to %f\n",
                  getDescription_c(),
                  mWeightUpdateTime);
         }
      }
      mLastUpdateTime = mInitialWeightUpdateTime;
   }

   return Response::SUCCESS;
}

Response::Status
HebbianUpdater::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = BaseWeightUpdater::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer = message->mDataRegistry;
   if (mPlasticityFlag and !mImmediateWeightUpdate) {
      mDeltaWeights->checkpointWeightPvp(checkpointer, name, "dW", mWriteCompressedCheckpoints);
      // Do we need to get PrepareCheckpointWrite messages, to call blockingNormalize_dW()?
   }
   std::string nameString = std::string(name);
   if (mPlasticityFlag && !mTriggerControl) {
      checkpointer->registerCheckpointData(
            nameString,
            "lastUpdateTime",
            &mLastUpdateTime,
            (std::size_t)1,
            true /*broadcast*/,
            false /*not constant*/);
      checkpointer->registerCheckpointData(
            nameString,
            "weightUpdateTime",
            &mWeightUpdateTime,
            (std::size_t)1,
            true /*broadcast*/,
            false /*not constant*/);
   }
   return Response::SUCCESS;
}

Response::Status HebbianUpdater::readStateFromCheckpoint(Checkpointer *checkpointer) {
   pvAssert(mInitializeFromCheckpointFlag);
   if (mPlasticityFlag and !mImmediateWeightUpdate) {
      checkpointer->readNamedCheckpointEntry(
            std::string(name), std::string("dW"), false /*not constant*/);
   }
   return Response::SUCCESS;
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
}

bool HebbianUpdater::needUpdate(double simTime, double dt) const {
   if (!mPlasticityFlag) {
      return false;
   }
   if (mTriggerControl) {
      return mTriggerControl->needUpdate(simTime + mTriggerOffset, dt);
   }
   else {
      double numUpdates = (simTime - mLastUpdateTime) / mWeightUpdatePeriod;
      double timeToClosest =
            std::fabs(numUpdates - std::nearbyint(numUpdates)) * mWeightUpdatePeriod;
      return timeToClosest < 0.5 * dt;
   }
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
   int const numArbors = mArborList->getNumAxonalArbors();
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
   int const numArbors      = mArborList->getNumAxonalArbors();
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
   int const numArbors        = mArborList->getNumAxonalArbors();
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
   HyPerLayer *pre           = mConnectionData->getPre();
   HyPerLayer *post          = mConnectionData->getPost();
   int nExt                  = pre->getNumExtended();
   PVLayerLoc const *preLoc  = pre->getLayerLoc();
   PVLayerLoc const *postLoc = post->getLayerLoc();
   int const nbatch          = preLoc->nbatch;
   int delay                 = mArborList->getDelay(arborID);

   float const *preactbufHead =
         pre->getComponentByType<BasePublisherComponent>()->getLayerData(delay);
   float const *postactbufHead = post->getComponentByType<BasePublisherComponent>()->getLayerData();

   if (mWeights->getSharedFlag()) {
      // Calculate x and y cell size
      int xCellSize  = zUnitCellSize(preLoc->nx, postLoc->nx);
      int yCellSize  = zUnitCellSize(preLoc->ny, postLoc->ny);
      int nxExt      = preLoc->nx + preLoc->halo.lt + preLoc->halo.rt;
      int nyExt      = preLoc->ny + preLoc->halo.up + preLoc->halo.dn;
      int nf         = preLoc->nf;
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
      HyPerLayer *clonePreLayer  = c->getPre();
      HyPerLayer *clonePostLayer = c->getPost();
      auto *clonePrePublisher    = clonePreLayer->getComponentByType<BasePublisherComponent>();
      auto *clonePostPublisher   = clonePostLayer->getComponentByType<BasePublisherComponent>();
      pvAssert(clonePrePublisher->getNumExtended() == nExt);
      pvAssert(clonePrePublisher->getLayerLoc()->nbatch == nbatch);
      float const *clonePre  = clonePrePublisher->getLayerData(delay);
      float const *clonePost = clonePostPublisher->getLayerData();
      for (int b = 0; b < nbatch; b++) {
         for (int kExt = 0; kExt < nExt; kExt++) {
            updateInd_dW(arborID, b, clonePre, clonePost, kExt);
         }
      }
   }

   return PV_SUCCESS;
}

void HebbianUpdater::updateInd_dW(
      int arborID,
      int batchID,
      float const *preLayerData,
      float const *postLayerData,
      int kExt) {
   HyPerLayer *pre           = mConnectionData->getPre();
   HyPerLayer *post          = mConnectionData->getPost();
   const PVLayerLoc *postLoc = post->getLayerLoc();

   const float *preactbuf  = preLayerData + batchID * pre->getNumExtended();
   const float *postactbuf = postLayerData + batchID * post->getNumExtended();

   int sya = (postLoc->nf * (postLoc->nx + postLoc->halo.lt + postLoc->halo.rt));

   float preact = preactbuf[kExt];
   if (preact == 0.0f) {
      return;
   }

   Patch const &patch = mWeights->getPatch(kExt);
   int ny             = patch.ny;
   int nk             = patch.nx * mWeights->getPatchSizeF();
   if (ny == 0 || nk == 0) {
      return;
   }

   size_t offset           = mWeights->getGeometry()->getAPostOffset(kExt);
   const float *postactRef = &postactbuf[offset];

   int sym = 0;

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
}

float HebbianUpdater::updateRule_dW(float pre, float post) { return mDWMax * pre * post; }

void HebbianUpdater::reduce_dW() {
   int status          = PV_SUCCESS;
   int const numArbors = mArborList->getNumAxonalArbors();
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
         FatalIf(
               kernel_status != activation_status,
               "%s reduce_dW returned differing values for reduceKernels() and "
               "reduceActivations()\n",
               getDescription_c());
      }
   }
   else {
      reduceAcrossBatch(arborId);
   }
   return kernel_status;
}

int HebbianUpdater::reduceKernels(int arborID) {
   pvAssert(mWeights->getSharedFlag() && mPlasticityFlag);
   Communicator const *comm = mCommunicator;
   const int nxProcs        = comm->numCommColumns();
   const int nyProcs        = comm->numCommRows();
   const int nbProcs        = comm->numCommBatches();
   const int nProcs         = nxProcs * nyProcs * nbProcs;
   if (nProcs != 1) {
      const MPI_Comm mpi_comm = comm->globalCommunicator();
      const int numPatches    = mWeights->getNumDataPatches();
      const size_t patchSize  = (size_t)mWeights->getPatchSizeOverall();
      const size_t localSize  = (size_t)numPatches * (size_t)patchSize;
      const size_t arborSize  = localSize * (size_t)mArborList->getNumAxonalArbors();

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
   Communicator const *comm = mCommunicator;
   const int nxProcs        = comm->numCommColumns();
   const int nyProcs        = comm->numCommRows();
   const int nbProcs        = comm->numCommBatches();
   const int nProcs         = nxProcs * nyProcs * nbProcs;
   if (mNumKernelActivations && nProcs != 1) {
      const MPI_Comm mpi_comm = comm->globalCommunicator();
      const int numPatches    = mWeights->getNumDataPatches();
      const size_t patchSize  = (size_t)mWeights->getPatchSizeOverall();
      const size_t localSize  = numPatches * patchSize;
      const size_t arborSize  = localSize * mArborList->getNumAxonalArbors();

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
   if (mCommunicator->numCommBatches() != 1) {
      const int numPatches     = mWeights->getNumDataPatches();
      const size_t patchSize   = (size_t)mWeights->getPatchSizeOverall();
      size_t const localSize   = (size_t)numPatches * (size_t)patchSize;
      size_t const arborSize   = localSize * (size_t)mArborList->getNumAxonalArbors();
      MPI_Comm const batchComm = mCommunicator->batchCommunicator();

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
      int const numArbors = mArborList->getNumAxonalArbors();
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
      int const numArbors  = mArborList->getNumAxonalArbors();
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
   int const numArbors = mArborList->getNumAxonalArbors();
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
   int const numArbors       = mArborList->getNumAxonalArbors();
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
   if (!mTriggerControl) {
      while (simTime >= mWeightUpdateTime) {
         mWeightUpdateTime += mWeightUpdatePeriod;
      }
   }
}

Response::Status HebbianUpdater::prepareCheckpointWrite() {
   blockingNormalize_dW();
   pvAssert(mDeltaWeightsReduceRequests.empty());
   return Response::SUCCESS;
}

Response::Status HebbianUpdater::cleanup() {
   if (!mDeltaWeightsReduceRequests.empty()) {
      wait_dWReduceRequests();
   }
   return Response::SUCCESS;
}

} // namespace PV
