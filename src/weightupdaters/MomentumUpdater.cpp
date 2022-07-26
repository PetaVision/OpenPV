/*
 * MomentumUpdater.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "MomentumUpdater.hpp"

#include "components/WeightsPair.hpp"
#include "io/LocalPatchWeightsFile.hpp"
#include "io/SharedWeightsFile.hpp"
#include <cmath> // exp()
#include <cstring> // memcpy(), strcmp()

namespace PV {

MomentumUpdater::MomentumUpdater(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void MomentumUpdater::initialize(char const *name, PVParams *params, Communicator const *comm) {
   HebbianUpdater::initialize(name, params, comm);
}

void MomentumUpdater::setObjectType() { mObjectType = "MomentumUpdater"; }

int MomentumUpdater::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HebbianUpdater::ioParamsFillGroup(ioFlag);
   ioParam_momentumMethod(ioFlag);
   ioParam_timeConstantTau(ioFlag);
   ioParam_momentumTau(ioFlag);
   ioParam_momentumDecay(ioFlag);
   ioParam_initPrev_dWFile(ioFlag);
   ioParam_prev_dWFrameNumber(ioFlag);
   return status;
}

void MomentumUpdater::ioParam_momentumMethod(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      parameters()->ioParamStringRequired(ioFlag, name, "momentumMethod", &mMomentumMethod);
      if (std::strcmp(mMomentumMethod, "viscosity") == 0) {
         mMethod = VISCOSITY;
      }
      else if (std::strcmp(mMomentumMethod, "simple") == 0) {
         mMethod = SIMPLE;
      }
      else if (std::strcmp(mMomentumMethod, "alex") == 0) {
         mMethod = ALEX;
      }
      else {
         Fatal() << "MomentumUpdater " << name << ": momentumMethod of " << mMomentumMethod
                 << " is not known. Options are \"viscosity\", \"simple\", and \"alex\".\n";
      }
   }
}

void MomentumUpdater::ioParam_timeConstantTau(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      pvAssert(!parameters()->presentAndNotBeenRead(name, "momentumMethod"));
      float defaultVal = 0;
      switch (mMethod) {
         case VISCOSITY: defaultVal = mDefaultTimeConstantTauViscosity; break;
         case SIMPLE: defaultVal    = mDefaultTimeConstantTauSimple; break;
         case ALEX: defaultVal      = mDefaultTimeConstantTauAlex; break;
         default: pvAssertMessage(0, "Unrecognized momentumMethod\n"); break;
      }

      // If momentumTau is being used instead of timeConstantTau, ioParam_momentumTau
      // will print a warning, so we don't warn if timeConstantTau is absent here.
      // When momentumTau is removed, warnIfAbsent should be set to true here.
      bool warnIfAbsent = !parameters()->present(getName(), "momentumTau");
      parameters()->ioParamValue(
            ioFlag, name, "timeConstantTau", &mTimeConstantTau, defaultVal, warnIfAbsent);
      if (ioFlag == PARAMS_IO_READ) {
         checkTimeConstantTau();
      }
   }
}

void MomentumUpdater::checkTimeConstantTau() {
   switch (mMethod) {
      case VISCOSITY:
         FatalIf(
               mTimeConstantTau < 0,
               "%s uses momentumMethod \"viscosity\" and so must have "
               "TimeConstantTau >= 0"
               " (value is %f).\n",
               getDescription_c(),
               (double)mTimeConstantTau);
         break;
      case SIMPLE:
         FatalIf(
               mTimeConstantTau < 0 or mTimeConstantTau >= 1,
               "%s uses momentumMethod \"simple\" and so must have "
               "TimeConstantTau >= 0 and timeConstantTau < 1"
               " (value is %f).\n",
               getDescription_c(),
               (double)mTimeConstantTau);
         break;
      case ALEX:
         FatalIf(
               mTimeConstantTau < 0 or mTimeConstantTau >= 1,
               "%s uses momentumMethod \"alex\" and so must have "
               "TimeConstantTau >= 0 and timeConstantTau < 1"
               " (value is %f).\n",
               getDescription_c(),
               (double)mTimeConstantTau);
         break;
      default: Fatal().printf("Unrecognized momentumMethod\n"); break;
   }
}

void MomentumUpdater::ioParam_momentumTau(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      pvAssert(!parameters()->presentAndNotBeenRead(name, "momentumMethod"));
      pvAssert(!parameters()->presentAndNotBeenRead(name, "timeConstantTau"));
      if (!parameters()->present(getName(), "momentumTau")) {
         return;
      }
      if (parameters()->present(getName(), "timeConstantTau")) {
         WarnLog().printf(
               "%s sets timeConstantTau, so momentumTau will be ignored.\n", getDescription_c());
         return;
      }
      mUsingDeprecatedMomentumTau = true;
      parameters()->ioParamValueRequired(ioFlag, name, "momentumTau", &mMomentumTau);
      WarnLog().printf(
            "%s uses momentumTau, which is deprecated. Use timeConstantTau instead.\n",
            getDescription_c());
   }
}

void MomentumUpdater::ioParam_momentumDecay(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      parameters()->ioParamValue(ioFlag, name, "momentumDecay", &mMomentumDecay, mMomentumDecay);
      if (mMomentumDecay < 0.0f || mMomentumDecay > 1.0f) {
         Fatal() << "MomentumUpdater " << name
                 << ": momentumDecay must be between 0 and 1 inclusive\n";
      }
   }
}

void MomentumUpdater::ioParam_initPrev_dWFile(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (mPlasticityFlag) {
      parameters()->ioParamString(
            ioFlag, name, "initPrev_dWFile", &mInitPrev_dWFile, "");
   }
}

void MomentumUpdater::ioParam_prev_dWFrameNumber(enum ParamsIOFlag ioFlag) {
   if (mPlasticityFlag) {
      pvAssert(!parameters()->presentAndNotBeenRead(name, "initPrev_dWFile"));
      if (mInitPrev_dWFile and mInitPrev_dWFile[0]) {
         parameters()->ioParamValue(
               ioFlag, name, "prev_dWFrameNumber", &mPrev_dWFrameNumber, mPrev_dWFrameNumber);
      }
   }
}

void MomentumUpdater::initMessageActionMap() {
   HebbianUpdater::initMessageActionMap();

   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<InitializeStateMessage const>(msgptr);
      return respondInitializeState(castMessage);
   };
   mMessageActionMap.emplace("InitializeState", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ConnectionOutputMessage const>(msgptr);
      return respondConnectionOutput(castMessage);
   };
   mMessageActionMap.emplace("ConnectionOutput", action);
}

Response::Status
MomentumUpdater::respondConnectionOutput(std::shared_ptr<ConnectionOutputMessage const> message) {
   outputMomentum(message->mTime);
   return Response::SUCCESS;
}

Response::Status
MomentumUpdater::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HebbianUpdater::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   
   auto *objectTable = message->mObjectTable;
   auto *weightsPair = objectTable->findObject<WeightsPair>(getName());
   mWriteStep = weightsPair->getWriteStep();
   mWriteTime = weightsPair->getInitialWriteTime();
   mWriteCompressedWeights = weightsPair->getWriteCompressedWeights();
   return Response::SUCCESS;
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

Response::Status
MomentumUpdater::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = HebbianUpdater::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   // Note: HebbianUpdater does not checkpoint dW if the mImmediateWeightUpdate flag is true.
   // Do we need to handle it here and in readStateFromCheckpoint? --pschultz, 2017-12-16
   if (mPlasticityFlag) {
      auto *checkpointer = message->mDataRegistry;
      mPrevDeltaWeights->checkpointWeightPvp(
            checkpointer, name, "prev_dW", mWriteCompressedCheckpoints);
      // Don't need to checkpoint the next write time because it will always be the same
      // as the WeightsPair's nextWrite.
      openOutputStateFile(message);
   }
   return Response::SUCCESS;
}

Response::Status
MomentumUpdater::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   Response::Status status = HebbianUpdater::initializeState(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (mPlasticityFlag and mInitPrev_dWFile and mInitPrev_dWFile[0]) {
      auto globalMPIBlock = getCommunicator()->getGlobalMPIBlock();
      auto fileManager = std::make_shared<FileManager>(globalMPIBlock, ".");
      std::shared_ptr<WeightsFile> weightsFile;
      if (mPrevDeltaWeights->getSharedFlag()) {
         weightsFile = std::make_shared<SharedWeightsFile>(
               fileManager,
               std::string(mInitPrev_dWFile),
               mPrevDeltaWeights->getData(),
               false /* compressedFlag */,
               true /* readOnlyFlag */,
               false /* clobberFlag */,
               false /* verifyWritesFlag */);
      }
      else {
         weightsFile = std::make_shared<LocalPatchWeightsFile>(
               fileManager,
               std::string(mInitPrev_dWFile),
               mPrevDeltaWeights->getData(),
               &mPrevDeltaWeights->getGeometry()->getPreLoc(),
               &mPrevDeltaWeights->getGeometry()->getPostLoc(),
               true /* fileExtendedFlag */,
               false /* compressedFlag */,
               true /* readOnlyFlag */,
               false /* clobberFlag */,
               false /* verifyWritesFlag */);
      }
      weightsFile->setIndex(mPrev_dWFrameNumber);
      weightsFile->read(*mPrevDeltaWeights->getData());
   }
   return status;
}

Response::Status MomentumUpdater::readStateFromCheckpoint(Checkpointer *checkpointer) {
   pvAssert(mInitializeFromCheckpointFlag);
   // Note: HebbianUpdater does not checkpoint dW if the mImmediateWeightUpdate flag is true.
   // Do we need to handle it here and in registerData? --pschultz, 2017-12-16
   if (mPlasticityFlag) {
      checkpointer->readNamedCheckpointEntry(
            std::string(name), std::string("prev_dW"), false /*not constant*/);
   }
   return Response::SUCCESS;
}

int MomentumUpdater::updateWeights(int arborId) {
   // Add momentum right before updateWeights
   if (mUsingDeprecatedMomentumTau) { // MomentumTau was deprecated Nov 19, 2018.
      WarnLog().printf(
            "%s is using momentumTau, which has been deprecated in favor of timeConstantTau.\n",
            getDescription_c());
      applyMomentumDeprecated(arborId);
   }
   else {
      applyMomentum(arborId);
   }

   // Current dW saved to prev_dW
   pvAssert(mPrevDeltaWeights);
   std::memcpy(
         mPrevDeltaWeights->getData(arborId),
         mDeltaWeights->getData(arborId),
         sizeof(float) * mDeltaWeights->getPatchSizeOverall() * mDeltaWeights->getNumDataPatches());

   // add dw to w
   return HebbianUpdater::updateWeights(arborId);
}

void MomentumUpdater::applyMomentum(int arborId) {
   // Shared weights done in parallel, parallel in numkernels
   float momentumFactor = mTimeConstantTau;
   switch (mMethod) {
      case VISCOSITY:
         momentumFactor = mTimeConstantTau ? std::exp(-1.0f / mTimeConstantTau) : 0.0f;
         applyMomentum(arborId, momentumFactor, mMomentumDecay);
         break;
      case SIMPLE: applyMomentum(arborId, momentumFactor, mMomentumDecay); break;
      case ALEX: applyMomentum(arborId, momentumFactor, mMomentumDecay * mDWMax); break;
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
         dwdata_start[k] *= 1 - dwFactor;
         dwdata_start[k] += dwFactor * prev_dw_start[k];
         dwdata_start[k] -= wFactor * wdata_start[k]; // TODO handle decay in a separate component
      }
   }
   // Since weights data is allocated with all patches of a given arbor in a single vector,
   // these two for-loops can probably be collapsed. --pschultz 2017-12-16
}

void MomentumUpdater::applyMomentumDeprecated(int arborId) {
   // Shared weights done in parallel, parallel in numkernels
   switch (mMethod) {
      case VISCOSITY:
         applyMomentumDeprecated(arborId, std::exp(-1.0f / mMomentumTau), mMomentumDecay);
         break;
      case SIMPLE: applyMomentumDeprecated(arborId, mMomentumTau, mMomentumDecay); break;
      case ALEX: applyMomentumDeprecated(arborId, mMomentumTau, mMomentumDecay * mDWMax); break;
      default: pvAssertMessage(0, "Unrecognized momentumMethod\n"); break;
   }
}

void MomentumUpdater::applyMomentumDeprecated(int arborId, float dwFactor, float wFactor) {
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

void MomentumUpdater::openOutputStateFile(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   if (mWriteStep < 0) { return; }

   auto *checkpointer = message->mDataRegistry;
   auto outputFileManager = getCommunicator()->getOutputFileManager();
   std::string outputStatePath(getName());
   outputStatePath.append(".prevDelta.pvp");

   // If the file exists and CheckpointReadDirectory is empty, we need to
   // clobber the file.
   if (checkpointer->getCheckpointReadDirectory().empty()) {
      outputFileManager->open(
            outputStatePath, std::ios_base::out, checkpointer->doesVerifyWrites());
   }

   auto *preLoc  = mConnectionData->getPre()->getLayerLoc();
   auto *postLoc = mConnectionData->getPost()->getLayerLoc();
   if (mPrevDeltaWeights->getSharedFlag()) {
      mWeightsFile = std::make_shared<SharedWeightsFile>(
            outputFileManager,
            outputStatePath,
            mPrevDeltaWeights->getData(),
            mWriteCompressedWeights,
            false /*readOnlyFlag*/,
            checkpointer->getCheckpointReadDirectory().empty() /*clobberFlag*/,
            checkpointer->doesVerifyWrites());
   }
   else {
      mWeightsFile = std::make_shared<LocalPatchWeightsFile>(
            outputFileManager,
            outputStatePath,
            mPrevDeltaWeights->getData(),
            preLoc,
            postLoc,
            true /*fileExtendedFlag*/,
            mWriteCompressedWeights,
            false /*readOnlyFlag*/,
            checkpointer->getCheckpointReadDirectory().empty() /*clobberFlag*/,
            checkpointer->doesVerifyWrites());
   }
   mWeightsFile->respond(message); // WeightsFile needs to register filepos
}

void MomentumUpdater::outputMomentum(double timestamp) {
   if (mWeightsFile && (timestamp >= mWriteTime)) {
      mWriteTime += mWriteStep;

      try {
         mWeightsFile->write(*mPrevDeltaWeights->getData(), timestamp);
      }
      catch (std::invalid_argument &e) {
         Fatal() << getDescription() << " unable to output momentum: " << e.what() << "\n";
      }
   }
   else if (mWriteStep < 0) {
      // If writeStep is negative, we never call writeWeights, but someone might restart from a
      // checkpoint with a different writeStep, so we maintain writeTime.
      mWriteTime = timestamp;
   }
}

} // namespace PV
