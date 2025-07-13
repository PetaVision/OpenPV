/*
 * MomentumUpdater.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "MomentumUpdater.hpp"

#include "components/WeightsPair.hpp"
#include "io/BroadcastPreWeightsFile.hpp"
#include "io/LocalPatchWeightsFile.hpp"
#include "io/SharedWeightsFile.hpp"
#include <cmath> // exp()
#include <cstring> // memcpy(), strcmp()

namespace PV {

MomentumUpdater::MomentumUpdater(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

void MomentumUpdater::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HebbianUpdater::initialize(params, defaults, comm);
}

void MomentumUpdater::setObjectType() { mObjectType = "MomentumUpdater"; }

int MomentumUpdater::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = HebbianUpdater::ioParamsFillGroup(ioSwitch);
   ioParam_momentumMethod(ioSwitch);
   ioParam_timeConstantTau(ioSwitch);
   ioParam_momentumTau(ioSwitch); // marked obsolete July 30, 2024
   ioParam_initPrev_dWFile(ioSwitch);
   ioParam_prev_dWFrameNumber(ioSwitch);
   return status;
}

void MomentumUpdater::ioParam_momentumMethod(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("plasticityFlag"));
   if (mPlasticityFlag) {
      mParamsIO->ioParam(ioSwitch, "momentumMethod", &mMomentumMethod);
      if (mMomentumMethod == "viscosity") {
         mMethod = VISCOSITY;
      }
      else if (mMomentumMethod == "simple") {
         mMethod = SIMPLE;
      }
      else if (mMomentumMethod == "alex") {
         // momentumMethod "alex" was marked obsolete on July 30, 2024
         Fatal().printf(
               "%s momentumMethod = \"alex\" is obsolete. "
               "Use \"viscosity\" or \"simple\" instead.\n",
               getDescription_c());
      }
      else {
         Fatal() << "MomentumUpdater " << getName() << ": momentumMethod of " << mMomentumMethod
                 << " is not known. Options are \"viscosity\" and \"simple\".\n";
      }
   }
}

void MomentumUpdater::ioParam_timeConstantTau(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("plasticityFlag"));
   if (mPlasticityFlag) {
      mParamsIO->ioParam(ioSwitch, "timeConstantTau", &mTimeConstantTau);
   }
}

void MomentumUpdater::checkTimeConstantTau() {
   switch (mMethod) {
      case VISCOSITY:
         FatalIf(
               mTimeConstantTau < 0,
               "%s uses momentumMethod \"viscosity\" and so must have "
               "TimeConstantTau >= 0 (value is %f).\n",
               getDescription_c(),
               (double)mTimeConstantTau);
         break;
      case SIMPLE:
         FatalIf(
               mTimeConstantTau < 0 or mTimeConstantTau >= 1,
               "%s uses momentumMethod \"simple\" and so must have "
               "TimeConstantTau >= 0 and timeConstantTau < 1 (value is %f).\n",
               getDescription_c(),
               (double)mTimeConstantTau);
         break;
      default: Fatal().printf("Unrecognized momentumMethod\n"); break;
   }
}

// momentumTau was marked obsolete on July 30, 2024.
void MomentumUpdater::ioParam_momentumTau(ParamsIOSwitch ioSwitch) {
   FatalIf(
         mParamsIO->isPresent("momentumTau"),
         "%s sets the momentumDecay parameter, which is obsolete. "
         "Use timeConstantTau instead.\n",
         getDescription_c());
}

void MomentumUpdater::ioParam_initPrev_dWFile(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("plasticityFlag"));
   if (mPlasticityFlag) {
      mParamsIO->ioParam(ioSwitch, "initPrev_dWFile", &mInitPrev_dWFile);
   }
}

void MomentumUpdater::ioParam_prev_dWFrameNumber(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("plasticityFlag"));
   if (mPlasticityFlag) {
      pvAssert(!mParamsIO->presentAndNotBeenRead("initPrev_dWFile"));
      if (!mInitPrev_dWFile.empty()) {
         mParamsIO->ioParam(ioSwitch, "prev_dWFrameNumber", &mPrev_dWFrameNumber);
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
   mPrevDeltaWeights = new Weights(getName(), mWeights);
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
            checkpointer, getName(), "prev_dW", mWriteCompressedCheckpoints);
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
   if (mPlasticityFlag and !mInitPrev_dWFile.empty()) {
      auto globalMPIBlock = getCommunicator()->getGlobalMPIBlock();
      char const *baseDirectory = mInitPrev_dWFile[0] == '/' ? "/" : ".";
      auto fileManager = std::make_shared<FileManager>(globalMPIBlock, baseDirectory);
      std::shared_ptr<WeightsFile> weightsFile;

      if (mPrevDeltaWeights->getSharedWeightsFlag()) {
         weightsFile = std::make_shared<SharedWeightsFile>(
               fileManager,
               mInitPrev_dWFile,
               mPrevDeltaWeights->getData(),
               false /* compressedFlag */,
               true /* readOnlyFlag */,
               false /* clobberFlag */,
               false /* verifyWritesFlag */);
      }
      else if (mPrevDeltaWeights->prelayerIsBroadcast()) {
         weightsFile = std::make_shared<BroadcastPreWeightsFile>(
               fileManager,
               mInitPrev_dWFile,
               mPrevDeltaWeights->getData(),
               mPrevDeltaWeights->getGeometry()->getPreLoc().nf,
               mPrevDeltaWeights->getGeometry()->getPostLoc().bcast,
               false /* compressedFlag */,
               true /*readOnlyFlag*/,
               false /*clobberFlag*/,
               false /* verifyWritesFlag */);
      }
      else {
         weightsFile = std::make_shared<LocalPatchWeightsFile>(
               fileManager,
               mInitPrev_dWFile,
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
      weightsFile->read();
   }
   return status;
}

Response::Status MomentumUpdater::readStateFromCheckpoint(Checkpointer *checkpointer) {
   pvAssert(mInitializeFromCheckpointFlag);
   // Note: HebbianUpdater does not checkpoint dW if the mImmediateWeightUpdate flag is true.
   // Do we need to handle it here and in registerData? --pschultz, 2017-12-16
   if (mPlasticityFlag) {
      checkpointer->readNamedCheckpointEntry(
            std::string(getName()), std::string("prev_dW"), false /*not constant*/);
   }
   return Response::SUCCESS;
}

int MomentumUpdater::updateWeights(int arborId) {
   // Add momentum right before updateWeights
   applyMomentum(arborId);

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
   float momentumFactor;
   if (mMethod == VISCOSITY) {
      momentumFactor = mTimeConstantTau ? std::exp(-1.0f / mTimeConstantTau) : 0.0f;
   }
   else {
      pvAssertMessage(mMethod == SIMPLE, "Unrecognized momentumMethod\n");
      momentumFactor = mTimeConstantTau;
   }
   applyMomentum(arborId, momentumFactor);
}

void MomentumUpdater::applyMomentum(int arborId, float dwFactor) {
   int const numKernels = mDeltaWeights->getNumDataPatches();
   pvAssert(numKernels == mPrevDeltaWeights->getNumDataPatches());
   int const patchSizeOverall = mDeltaWeights->getPatchSizeOverall();
   pvAssert(patchSizeOverall == mPrevDeltaWeights->getPatchSizeOverall());
   auto deltaWeightData = mDeltaWeights->getData();
   auto prevDeltaWeightData = mPrevDeltaWeights->getData();
   auto weightData = mWeights->getData();
   long int numValuesPerArbor = weightData->getNumValuesPerArbor();
   pvAssert(deltaWeightData->getNumValuesPerArbor() == numValuesPerArbor);
   pvAssert(prevDeltaWeightData->getNumValuesPerArbor() == numValuesPerArbor);
   float *dwdata_start        = deltaWeightData->getData(arborId);
   float const *prev_dw_start = prevDeltaWeightData->getData(arborId);
   float const *wdata_start   = weightData->getData(arborId);
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for (int k = 0; k < numValuesPerArbor; ++k) {
      float dw = (1 - dwFactor) * dwdata_start[k] + dwFactor * prev_dw_start[k];
      dwdata_start[k] = dw;
   }
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
   if (mPrevDeltaWeights->getSharedWeightsFlag()) {
      mWeightsFile = std::make_shared<SharedWeightsFile>(
            outputFileManager,
            outputStatePath,
            mPrevDeltaWeights->getData(),
            mWriteCompressedWeights,
            false /*readOnlyFlag*/,
            checkpointer->getCheckpointReadDirectory().empty() /*clobberFlag*/,
            checkpointer->doesVerifyWrites());
   }
   else if (mPrevDeltaWeights->prelayerIsBroadcast()) {
      mWeightsFile = std::make_shared<BroadcastPreWeightsFile>(
            outputFileManager,
            outputStatePath,
            mPrevDeltaWeights->getData(),
            preLoc->nf,
            postLoc->bcast,
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
         mWeightsFile->write(timestamp);
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
