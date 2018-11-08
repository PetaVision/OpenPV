/*
 * MomentumConnSimpleCheckpointerTestProbe.cpp
 *
 *  Created on: Jan 6, 2017
 *      Author: pschultz
 */

#include "MomentumConnSimpleCheckpointerTestProbe.hpp"
#include "components/ArborList.hpp"
#include "components/InputActivityBuffer.hpp"
#include "components/PatchSize.hpp"
#include "components/SharedWeights.hpp"
#include "components/WeightsPair.hpp"
#include "weightupdaters/MomentumUpdater.hpp"
#include <cmath>
#include <utils/BufferUtilsMPI.hpp>

MomentumConnSimpleCheckpointerTestProbe::MomentumConnSimpleCheckpointerTestProbe() {}

MomentumConnSimpleCheckpointerTestProbe::MomentumConnSimpleCheckpointerTestProbe(
      const char *name,
      PV::PVParams *params,
      PV::Communicator *comm) {
   initialize(name, params, comm);
}

MomentumConnSimpleCheckpointerTestProbe::~MomentumConnSimpleCheckpointerTestProbe() {}

void MomentumConnSimpleCheckpointerTestProbe::initialize(
      const char *name,
      PV::PVParams *params,
      PV::Communicator *comm) {
   return PV::ColProbe::initialize(name, params, comm);
}

void MomentumConnSimpleCheckpointerTestProbe::ioParam_textOutputFlag(enum PV::ParamsIOFlag ioFlag) {
   ColProbe::ioParam_textOutputFlag(ioFlag);
   if (ioFlag == PV::PARAMS_IO_READ && !getTextOutputFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         ErrorLog() << getDescription() << ": MomentumConnSimpleCheckpointerTestProbe requires "
                                           "textOutputFlag to be set to true.\n";
      }
   }
}

PV::Response::Status MomentumConnSimpleCheckpointerTestProbe::communicateInitInfo(
      std::shared_ptr<PV::CommunicateInitInfoMessage const> message) {
   auto status = PV::ColProbe::communicateInitInfo(message);
   if (!PV::Response::completed(status)) {
      return status;
   }

   auto *componentTable = message->mHierarchy;
   status               = status + initInputLayer(componentTable);
   status               = status + initOutputLayer(componentTable);
   status               = status + initConnection(componentTable);

   if (PV::Response::completed(status)) {
      mInitializeFromCheckpointFlag = mConnection->getInitializeFromCheckpointFlag();
      FatalIf(
            mInputLayer->getInitializeFromCheckpointFlag() != mInitializeFromCheckpointFlag,
            "%s has a different initializeFromCheckpointFlag value from the connection %s.\n",
            mInputLayer->getDescription(),
            mConnection->getDescription());
      FatalIf(
            mOutputLayer->getInitializeFromCheckpointFlag() != mInitializeFromCheckpointFlag,
            "%s has a different initializeFromCheckpointFlag value from the connection %s.\n",
            mOutputLayer->getDescription(),
            mConnection->getDescription());
   }

   return status;
}

PV::Response::Status
MomentumConnSimpleCheckpointerTestProbe::initInputLayer(PV::ObserverTable const *componentTable) {
   mInputLayer = componentTable->lookupByName<PV::InputLayer>(std::string("Input"));
   FatalIf(mInputLayer == nullptr, "column does not have an InputLayer named \"Input\".\n");
   if (checkCommunicatedFlag(mInputLayer) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }

   auto *activityComponent = mInputLayer->getComponentByType<PV::ActivityComponent>();
   auto *inputBuffer       = activityComponent->getComponentByType<PV::InputActivityBuffer>();
   FatalIf(
         inputBuffer->getDisplayPeriod() != 4.0,
         "This test assumes that the display period is 4 (should really not be hard-coded.\n");
   return PV::Response::SUCCESS;
}

PV::Response::Status
MomentumConnSimpleCheckpointerTestProbe::initOutputLayer(PV::ObserverTable const *componentTable) {
   mOutputLayer = componentTable->lookupByName<PV::HyPerLayer>(std::string("Output"));
   FatalIf(mOutputLayer == nullptr, "column does not have a HyPerLayer named \"Output\".\n");
   if (checkCommunicatedFlag(mOutputLayer) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }
   return PV::Response::SUCCESS;
}

PV::Response::Status
MomentumConnSimpleCheckpointerTestProbe::initConnection(PV::ObserverTable const *componentTable) {
   mConnection =
         componentTable->lookupByName<PV::ComponentBasedObject>(std::string("InputToOutput"));
   FatalIf(
         mConnection == nullptr, "column does not have a MomentumConn named \"InputToOutput\".\n");
   if (checkCommunicatedFlag(mConnection) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }

   auto *arborList = mConnection->getComponentByType<PV::ArborList>();
   FatalIf(
         arborList == nullptr, "%s does not have an ArborList.\n", mConnection->getDescription_c());
   FatalIf(
         arborList->getNumAxonalArbors() != 1,
         "This test assumes that the connection has only 1 arbor.\n");
   FatalIf(
         arborList->getDelay(0) != 0.0, "This test assumes that the connection has zero delay.\n");

   auto *sharedWeights = mConnection->getComponentByType<PV::SharedWeights>();
   FatalIf(
         sharedWeights == nullptr,
         "%s does not have a SharedWeights component.\n",
         mConnection->getDescription_c());
   FatalIf(
         !sharedWeights->getSharedWeights(),
         "This test assumes that the connection is using shared weights.\n");

   auto *patchSize = mConnection->getComponentByType<PV::PatchSize>();
   FatalIf(
         patchSize == nullptr,
         "%s does not have a PatchSize component.\n",
         mConnection->getDescription_c());
   FatalIf(patchSize->getPatchSizeX() != 1, "This test assumes that the connection has nxp==1.\n");
   FatalIf(patchSize->getPatchSizeY() != 1, "This test assumes that the connection has nyp==1.\n");
   FatalIf(patchSize->getPatchSizeF() != 1, "This test assumes that the connection has nfp==1.\n");

   auto *momentumUpdater = mConnection->getComponentByType<PV::MomentumUpdater>();
   FatalIf(
         momentumUpdater == nullptr,
         "%s does not have a momentumUpdater.\n",
         mConnection->getDescription_c());
   FatalIf(
         std::strcmp(momentumUpdater->getMomentumMethod(), "simple"),
         "This test assumes that the connection has momentumMethod=\"simple\".\n");
   return PV::Response::SUCCESS;
}

PV::Response::Status
MomentumConnSimpleCheckpointerTestProbe::checkCommunicatedFlag(PV::BaseObject *dependencyObject) {
   if (!dependencyObject->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->commRank() == 0) {
         InfoLog().printf(
               "%s must wait until \"%s\" has finished its communicateInitInfo stage.\n",
               getDescription_c(),
               dependencyObject->getName());
      }
      return PV::Response::POSTPONE;
   }
   else {
      return PV::Response::SUCCESS;
   }
}

PV::Response::Status MomentumConnSimpleCheckpointerTestProbe::initializeState(
      std::shared_ptr<PV::InitializeStateMessage const> message) {
   FatalIf(message->mDeltaTime != 1.0, "This test assumes that the HyPerCol dt is 1.0.\n");
   return PV::Response::SUCCESS;
}

PV::Response::Status
MomentumConnSimpleCheckpointerTestProbe::readStateFromCheckpoint(PV::Checkpointer *checkpointer) {
   PV::Checkpointer::TimeInfo timeInfo;
   PV::CheckpointEntryData<PV::Checkpointer::TimeInfo> timeInfoCheckpointEntry(
         std::string("timeinfo"),
         mCommunicator->getLocalMPIBlock(),
         &timeInfo,
         (size_t)1,
         true /*broadcast*/);
   std::string initializeFromCheckpointDir(checkpointer->getInitializeFromCheckpointDir());
   timeInfoCheckpointEntry.read(initializeFromCheckpointDir, nullptr);

   mStartingUpdateNumber = calcUpdateNumber(timeInfo.mSimTime);

   return PV::Response::SUCCESS;
}

int MomentumConnSimpleCheckpointerTestProbe::calcUpdateNumber(double timevalue) {
   pvAssert(timevalue >= 0.0);
   int const step = (int)std::nearbyint(timevalue);
   pvAssert(step >= 0);
   int const updateNumber = (step + 3) / 4; // integer division
   return updateNumber;
}

void MomentumConnSimpleCheckpointerTestProbe::initializeCorrectValues(double timevalue) {
   int const updateNumber = mStartingUpdateNumber + calcUpdateNumber(timevalue);
   if (updateNumber == 0) {
      mCorrectState =
            new CorrectState(0, 1.0f /*weight*/, 0.0f /*dw*/, 1.0f /*input*/, 2.0f /*output*/);
   }
   else {
      mCorrectState =
            new CorrectState(1, 3.0f /*weight*/, 2.0f /*dw*/, 1.0f /*input*/, 3.0f /*output*/);

      for (int j = 2; j < updateNumber; j++) {
         mCorrectState->update();
      }
      // Don't update for the current updateNumber; this will happen in outputState.
   }
}

PV::Response::Status
MomentumConnSimpleCheckpointerTestProbe::outputState(double simTime, double deltaTime) {
   if (!mValuesSet) {
      initializeCorrectValues(simTime);
      mValuesSet = true;
   }
   int const updateNumber = mStartingUpdateNumber + calcUpdateNumber(simTime);
   while (updateNumber > mCorrectState->getUpdateNumber()) {
      mCorrectState->update();
   }

   bool failed = false;

   failed |= verifyConnection(mConnection, mCorrectState, simTime);
   failed |= verifyLayer(mInputLayer, mCorrectState->getCorrectInput(), simTime);
   failed |= verifyLayer(mOutputLayer, mCorrectState->getCorrectOutput(), simTime);

   if (failed) {
      std::string errorMsg(getDescription() + " failed at t = " + std::to_string(simTime) + "\n");
      if (!mOutputStreams.empty()) {
         output(0).printf(errorMsg.c_str());
      }
      if (isWritingToFile()) { // print error message to screen/log file as well.
         ErrorLog() << errorMsg;
      }
      mTestFailed = true;
   }
   else {
      if (!mOutputStreams.empty()) {
         output(0).printf("%s found all correct values at time %f\n", getDescription_c(), simTime);
      }
   }
   // Test runs all timesteps and then checks the mTestFailed flag at the end.
   return PV::Response::SUCCESS;
}

bool MomentumConnSimpleCheckpointerTestProbe::verifyConnection(
      PV::ComponentBasedObject *connection,
      CorrectState const *correctState,
      double timevalue) {
   bool failed = false;

   if (mCommunicator->commRank() == 0) {
      auto *weightsPair         = connection->getComponentByType<PV::WeightsPair>();
      float observedWeightValue = weightsPair->getPreWeights()->getData(0)[0];
      float correctWeightValue  = correctState->getCorrectWeight();
      failed |= verifyConnValue(timevalue, observedWeightValue, correctWeightValue, "weight");

      auto *updater = connection->getComponentByType<PV::MomentumUpdater>();
      pvAssert(updater);
      float observed_dwValue = updater->getDeltaWeightsDataStart(0)[0];
      float correct_dwValue  = correctState->getCorrect_dw();
      failed |= verifyConnValue(timevalue, observed_dwValue, correct_dwValue, "dw");
   }
   return failed;
}

bool MomentumConnSimpleCheckpointerTestProbe::verifyConnValue(
      double timevalue,
      float observed,
      float correct,
      char const *valueDescription) {
   FatalIf(
         mCommunicator->commRank() != 0,
         "%s called verifyConnValue from nonroot process.\n",
         getDescription_c());
   FatalIf(
         mOutputStreams.empty(),
         "%s has empty mOutputStreams in root process.\n",
         getDescription_c());
   bool failed;
   if (observed != correct) {
      output(0).printf(
            "Time %f, %s is %f, instead of the expected %f.\n",
            timevalue,
            valueDescription,
            (double)observed,
            (double)correct);
      failed = true;
   }
   else {
      failed = false;
   }
   return failed;
}

bool MomentumConnSimpleCheckpointerTestProbe::verifyLayer(
      PV::HyPerLayer *layer,
      float correctValue,
      double timevalue) {
   bool failed = false;

   PVLayerLoc const *inputLoc = layer->getLayerLoc();
   PVHalo const *inputHalo    = &inputLoc->halo;
   int const inputNxExt       = inputLoc->nx + inputHalo->lt + inputHalo->rt;
   int const inputNyExt       = inputLoc->ny + inputHalo->dn + inputHalo->up;
   PV::Buffer<float> localBuffer(layer->getLayerData(0), inputNxExt, inputNyExt, inputLoc->nf);
   PV::Communicator *comm         = mCommunicator;
   PV::Buffer<float> globalBuffer = PV::BufferUtils::gather(
         comm->getLocalMPIBlock(), localBuffer, inputLoc->nx, inputLoc->ny, 0, 0);
   if (comm->commRank() == 0) {
      FatalIf(
            mOutputStreams.empty(),
            "%s has empty mOutputStreams in root process.\n",
            getDescription_c());
      globalBuffer.crop(inputLoc->nxGlobal, inputLoc->nyGlobal, PV::Buffer<float>::CENTER);
      std::vector<float> globalVector = globalBuffer.asVector();
      int const numInputNeurons       = globalVector.size();
      for (int k = 0; k < numInputNeurons; k++) {
         if (globalVector[k] != correctValue) {
            output(0).printf(
                  "Time %f, %s neuron %d is %f, instead of the expected %f.\n",
                  timevalue,
                  layer->getName(),
                  k,
                  (double)globalVector[k],
                  (double)correctValue);
            failed = true;
         }
      }
   }
   return failed;
}
