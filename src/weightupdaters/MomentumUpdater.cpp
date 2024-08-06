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
   ioParam_momentumTau(ioFlag); // marked obsolete July 30, 2024
   ioParam_momentumDecay(ioFlag); // deprecated July 30, 2024
   ioParam_weightL1Decay(ioFlag);
   ioParam_weightL2Decay(ioFlag);
   ioParam_initPrev_dWFile(ioFlag);
   ioParam_prev_dWFrameNumber(ioFlag);
   return status;
}

void MomentumUpdater::ioParam_momentumMethod(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "plasticityFlag"));
   if (mPlasticityFlag) {
      parameters()->ioParamStringRequired(ioFlag, getName(), "momentumMethod", &mMomentumMethod);
      if (std::strcmp(mMomentumMethod, "viscosity") == 0) {
         mMethod = VISCOSITY;
      }
      else if (std::strcmp(mMomentumMethod, "simple") == 0) {
         mMethod = SIMPLE;
      }
      else if (std::strcmp(mMomentumMethod, "alex") == 0) {
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

void MomentumUpdater::ioParam_timeConstantTau(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "plasticityFlag"));
   if (mPlasticityFlag) {
      pvAssert(!parameters()->presentAndNotBeenRead(getName(), "momentumMethod"));
      float defaultVal = 0;
      switch (mMethod) {
         case VISCOSITY: defaultVal = mDefaultTimeConstantTauViscosity; break;
         case SIMPLE: defaultVal    = mDefaultTimeConstantTauSimple; break;
         default: pvAssertMessage(0, "Unrecognized momentumMethod\n"); break;
      }

      bool warnIfAbsent = true;
      parameters()->ioParamValue(
            ioFlag, getName(), "timeConstantTau", &mTimeConstantTau, defaultVal, warnIfAbsent);
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

// momentumTau was marked obsolete on July 30, 2025.
void MomentumUpdater::ioParam_momentumTau(enum ParamsIOFlag ioFlag) {
   FatalIf(
         parameters()->present(getName(), "momentumTau"),
         "%s sets the momentumDecay parameter, which is obsolete. "
         "Use timeConstantTau instead.\n",
         getDescription_c());
}

// momentumDecay was deprecated on July 30, 2025.
void MomentumUpdater::ioParam_momentumDecay(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "plasticityFlag"));
   if (mPlasticityFlag) {
      if (ioFlag == PARAMS_IO_READ and parameters()->present(getName(), "momentumDecay")) {
         WarnLog().printf(
               "%s sets momentumDecay parameter, which is deprecated. Use weightL2Decay instead.",
               getDescription_c());
         if (parameters()->present(getName(), "weightL2Decay")) {
            return; // ioParam_weightL2Decay() will handle it
         }
         else {
            parameters()->ioParamValue(
                 ioFlag, getName(), "momentumDecay", &mWeightL2Decay, mWeightL2Decay);
            if (mWeightL2Decay < 0.0f || mWeightL2Decay > 1.0f) {
               Fatal() << "MomentumUpdater " << getName()
                       << ": weightL2Decay must be between 0 and 1 inclusive\n";
            }
         }
      }
   }
}

// Once momentumDecay is marked obsolete, this function can be reduced to
// pvAssert(!parameters()->presentAndNotBeenRead(getName(), "plasticityFlag"));
// if (mPlasticityFlag) {
//    parameters()->ioParamValue(
//          ioFlag, getName(), "weightL2Decay", &mWeightL2Decay, mWeightL2Decay);
//    if (mWeightL2Decay < 0.0f || mWeightL2Decay > 1.0f) {
//       Fatal() << getDescription_c()
//               << ": weightL2Decay must be between 0 and 1 inclusive\n";
//    }
// }
void MomentumUpdater::ioParam_weightL2Decay(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "plasticityFlag"));
   if (mPlasticityFlag) {
      if (ioFlag == PARAMS_IO_READ) {
         pvAssert(!parameters()->presentAndNotBeenRead(getName(), "momentumDecay"));
         bool usesMomentumDecay = parameters()->present(getName(), "momentumDecay");
         bool usesWeightL2Decay = parameters()->present(getName(), "weightL2Decay");
         if (usesMomentumDecay and !usesWeightL2Decay) {
            return; // ioParam_momentumDecay() read the parameter into mWeightL2Decay already.
         }
         parameters()->ioParamValue(
               ioFlag, getName(), "weightL2Decay", &mWeightL2Decay, mWeightL2Decay);
         if (mWeightL2Decay < 0.0f || mWeightL2Decay > 1.0f) {
            Fatal() << getDescription_c()
                    << ": weightL2Decay must be between 0 and 1 inclusive\n";
         }
         FatalIf(
               mWeightL2Decay < 0.0f or mWeightL2Decay > 1.0f,
               "%s: weightL2Decay must be between 0 and 1 inclusive (given value was %f)\n",
               getDescription_c(),
               static_cast<double>(mWeightL2Decay));
      }
      else { // PARAMS_IO_WRITE
         parameters()->ioParamValue(
               ioFlag, getName(), "weightL2Decay", &mWeightL2Decay, mWeightL2Decay);
      }
   }
}

void MomentumUpdater::ioParam_weightL1Decay(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "plasticityFlag"));
   if (mPlasticityFlag) {
      parameters()->ioParamValue(
               ioFlag, getName(), "weightL1Decay", &mWeightL1Decay, mWeightL1Decay);
      FatalIf(
            mWeightL1Decay < 0.0f,
            "%s: weightL1Decay cannot be negative (given value was %f)\n",
            getDescription_c(),
            static_cast<double>(mWeightL1Decay));
   }
}

void MomentumUpdater::ioParam_initPrev_dWFile(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(getName(), "plasticityFlag"));
   if (mPlasticityFlag) {
      parameters()->ioParamString(
            ioFlag, getName(), "initPrev_dWFile", &mInitPrev_dWFile, "");
   }
}

void MomentumUpdater::ioParam_prev_dWFrameNumber(enum ParamsIOFlag ioFlag) {
   if (mPlasticityFlag) {
      pvAssert(!parameters()->presentAndNotBeenRead(getName(), "initPrev_dWFile"));
      if (mInitPrev_dWFile and mInitPrev_dWFile[0]) {
         parameters()->ioParamValue(
               ioFlag, getName(), "prev_dWFrameNumber", &mPrev_dWFrameNumber, mPrev_dWFrameNumber);
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
   if (mPlasticityFlag and mInitPrev_dWFile and mInitPrev_dWFile[0]) {
      auto globalMPIBlock = getCommunicator()->getGlobalMPIBlock();
      char const *baseDirectory = mInitPrev_dWFile[0] == '/' ? "/" : ".";
      auto fileManager = std::make_shared<FileManager>(globalMPIBlock, baseDirectory);
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
      float dw = dwdata_start[k];
      dw *= 1 - dwFactor;
      dw += dwFactor * prev_dw_start[k];
      float weight  = wdata_start[k];
      float decayL2 = mWeightL2Decay * weight;
      float dwL1    = mWeightL1Decay;
      float decayL1 = dwL1 * ((weight > dwL1) - (weight < -dwL1));
      decayL1 += weight * (std::abs(weight) <= dwL1);
      // Formula for decayL1 is = mWeightL1Decay * sgn(w) if |w| > mWeightL1Decay;
      //                          |w| if |w| <= mWeightL1Decay
      dw -= decayL2 + decayL1;
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
