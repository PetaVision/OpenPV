/*
 * MomentumConnViscosityCheckpointerTestProbe.cpp
 *
 *  Created on: Jan 6, 2017
 *      Author: pschultz
 */

#include "MomentumConnViscosityCheckpointerTestProbe.hpp"
#include "components/ArborList.hpp"
#include "components/InputActivityBuffer.hpp"
#include "components/PatchSize.hpp"
#include "components/SharedWeights.hpp"
#include "components/WeightsPair.hpp"
#include "weightupdaters/MomentumUpdater.hpp"
#include <cmath>
#include <utils/BufferUtilsMPI.hpp>

MomentumConnViscosityCheckpointerTestProbe::MomentumConnViscosityCheckpointerTestProbe() {}

MomentumConnViscosityCheckpointerTestProbe::MomentumConnViscosityCheckpointerTestProbe(
      const char *name,
      PV::PVParams *params,
      PV::Communicator const *comm) {
   initialize(name, params, comm);
}

MomentumConnViscosityCheckpointerTestProbe::~MomentumConnViscosityCheckpointerTestProbe() {}

void MomentumConnViscosityCheckpointerTestProbe::initialize(
      const char *name,
      PV::PVParams *params,
      PV::Communicator const *comm) {
   return PV::ColProbe::initialize(name, params, comm);
}

void MomentumConnViscosityCheckpointerTestProbe::ioParam_textOutputFlag(
      enum PV::ParamsIOFlag ioFlag) {
   ColProbe::ioParam_textOutputFlag(ioFlag);
   if (ioFlag == PV::PARAMS_IO_READ && !getTextOutputFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         ErrorLog() << getDescription() << ": MomentumConnViscosityCheckpointerTestProbe requires "
                                           "textOutputFlag to be set to true.\n";
      }
   }
}

PV::Response::Status MomentumConnViscosityCheckpointerTestProbe::communicateInitInfo(
      std::shared_ptr<PV::CommunicateInitInfoMessage const> message) {
   auto status = PV::ColProbe::communicateInitInfo(message);
   if (!PV::Response::completed(status)) {
      return status;
   }

   auto *objectTable = message->mObjectTable;

   status = mConnection ? status : status + initConnection(objectTable);
   // initConnection sets InitializeFromCheckpointFlag, and init{In,Out}PutLayer checks
   // against that value, so we have to complete initConnection successively before
   // calling initInputLayer or initOutputLayer.
   if (!PV::Response::completed(status)) {
      return status;
   }
   status = mInputPublisher ? status : status + initInputLayer(objectTable);
   status = mOutputPublisher ? status : status + initOutputLayer(objectTable);

   return status;
}

PV::Response::Status
MomentumConnViscosityCheckpointerTestProbe::initConnection(PV::ObserverTable const *objectTable) {
   char const *connectionName = "InputToOutput";
   auto *connection           = objectTable->findObject<PV::ComponentBasedObject>(connectionName);
   FatalIf(
         connection == nullptr,
         "column does not have a MomentumConn named \"%s\".\n",
         connectionName);
   if (checkCommunicatedFlag(connection) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }
   mConnection = connection;

   auto *arborList = objectTable->findObject<PV::ArborList>(connectionName);
   FatalIf(arborList == nullptr, "Connection \"%s\" does not have an ArborList.\n", connectionName);
   if (checkCommunicatedFlag(arborList) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }
   FatalIf(
         arborList->getNumAxonalArbors() != 1,
         "This test assumes that the connection has only 1 arbor.\n");
   FatalIf(
         arborList->getDelay(0) != 0.0, "This test assumes that the connection has zero delay.\n");

   auto *sharedWeights = objectTable->findObject<PV::SharedWeights>(connectionName);
   FatalIf(
         sharedWeights == nullptr,
         "%s does not have a SharedWeights component.\n",
         mConnection->getDescription_c());
   if (checkCommunicatedFlag(sharedWeights) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }
   FatalIf(
         !sharedWeights->getSharedWeights(),
         "This test assumes that the connection is using shared weights.\n");

   auto *patchSize = objectTable->findObject<PV::PatchSize>(connectionName);
   FatalIf(
         patchSize == nullptr,
         "%s does not have a PatchSize component.\n",
         mConnection->getDescription_c());
   if (checkCommunicatedFlag(patchSize) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }
   FatalIf(patchSize->getPatchSizeX() != 1, "This test assumes that the connection has nxp==1.\n");
   FatalIf(patchSize->getPatchSizeY() != 1, "This test assumes that the connection has nyp==1.\n");
   FatalIf(patchSize->getPatchSizeF() != 1, "This test assumes that the connection has nfp==1.\n");

   auto *momentumUpdater = objectTable->findObject<PV::MomentumUpdater>(connectionName);
   FatalIf(
         momentumUpdater == nullptr,
         "%s does not have a momentumUpdater.\n",
         mConnection->getDescription_c());
   if (checkCommunicatedFlag(momentumUpdater) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }
   FatalIf(
         std::strcmp(momentumUpdater->getMomentumMethod(), "viscosity"),
         "This test assumes that the connection has momentumMethod=\"viscosity\".\n");
   mInitializeFromCheckpointFlag = mConnection->getInitializeFromCheckpointFlag();
   return PV::Response::SUCCESS;
}

PV::Response::Status
MomentumConnViscosityCheckpointerTestProbe::initInputLayer(PV::ObserverTable const *objectTable) {
   char const *inputLayerName = "Input";
   auto *inputLayer           = objectTable->findObject<PV::InputLayer>(inputLayerName);
   FatalIf(
         inputLayer == nullptr,
         "column does not have an InputLayer named \"%s\".\n",
         inputLayerName);
   if (checkCommunicatedFlag(inputLayer) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }
   FatalIf(
         inputLayer->getInitializeFromCheckpointFlag() != mInitializeFromCheckpointFlag,
         "%s has a different initializeFromCheckpointFlag value from the probe %s.\n",
         inputLayer->getDescription_c(),
         getDescription_c());

   auto *inputBuffer = objectTable->findObject<PV::InputActivityBuffer>(inputLayerName);
   FatalIf(
         inputBuffer == nullptr,
         "%s does not have an InputActivityBuffer.\n",
         inputLayer->getDescription_c());
   if (checkCommunicatedFlag(inputBuffer) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }
   FatalIf(
         inputBuffer->getDisplayPeriod() != 4.0,
         "This test assumes that the display period is 4 (should really not be hard-coded.\n");

   mInputPublisher = objectTable->findObject<PV::BasePublisherComponent>(inputLayerName);
   FatalIf(
         mInputPublisher == nullptr,
         "%s does not have a BasePublisherComponent.\n",
         inputLayer->getDescription_c());
   return PV::Response::SUCCESS;
}

PV::Response::Status
MomentumConnViscosityCheckpointerTestProbe::initOutputLayer(PV::ObserverTable const *objectTable) {
   char const *outputLayerName = "Output";
   auto *outputLayer           = objectTable->findObject<PV::HyPerLayer>(outputLayerName);
   FatalIf(
         outputLayer == nullptr,
         "column does not have a HyPerLayer named \"%s\".\n",
         outputLayerName);
   if (checkCommunicatedFlag(outputLayer) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }
   FatalIf(
         outputLayer->getInitializeFromCheckpointFlag() != mInitializeFromCheckpointFlag,
         "%s has a different initializeFromCheckpointFlag value from the probe %s.\n",
         outputLayer->getDescription_c(),
         getDescription_c());
   mOutputPublisher = objectTable->findObject<PV::BasePublisherComponent>(outputLayerName);
   if (checkCommunicatedFlag(mOutputPublisher) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }
   FatalIf(
         mOutputPublisher == nullptr,
         "%s does not have a BasePublisherComponent.\n",
         outputLayer->getDescription_c());
   return PV::Response::SUCCESS;
}

PV::Response::Status MomentumConnViscosityCheckpointerTestProbe::checkCommunicatedFlag(
      PV::BaseObject *dependencyObject) {
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

PV::Response::Status MomentumConnViscosityCheckpointerTestProbe::initializeState(
      std::shared_ptr<PV::InitializeStateMessage const> message) {
   FatalIf(message->mDeltaTime != 1.0, "This test assumes that the HyPerCol dt is 1.0.\n");
   return PV::Response::SUCCESS;
}

PV::Response::Status MomentumConnViscosityCheckpointerTestProbe::readStateFromCheckpoint(
      PV::Checkpointer *checkpointer) {
   PV::Checkpointer::TimeInfo timeInfo;
   PV::CheckpointEntryData<PV::Checkpointer::TimeInfo> timeInfoCheckpointEntry(
         std::string("timeinfo"),
         mCommunicator->getLocalMPIBlock(),
         &timeInfo,
         (size_t)1,
         true /*broadcast*/);
   std::string initializeFromCheckpointDir(checkpointer->getInitializeFromCheckpointDir());
   timeInfoCheckpointEntry.read(initializeFromCheckpointDir, nullptr);

   mStartingTimestamp = timeInfo.mSimTime;

   return PV::Response::SUCCESS;
}

void MomentumConnViscosityCheckpointerTestProbe::initializeCorrectValues() {
   auto *momentumUpdater = mConnection->getComponentByType<PV::MomentumUpdater>();
   float const tau       = momentumUpdater->getTimeConstantTau();
   float const tau_exp   = std::exp(-1.0f / tau);
   mCorrectState =
         new CorrectState(tau_exp, 1.0f /*weight*/, 0.0f /*dw*/, 1.0f /*input*/, 1.0f /*output*/);
}

PV::Response::Status
MomentumConnViscosityCheckpointerTestProbe::outputState(double simTime, double deltaTime) {
   if (!mValuesSet) {
      initializeCorrectValues();
      mValuesSet = true;
   }
   int const updateNumber = mStartingTimestamp + simTime;
   while (updateNumber > mCorrectState->getTimestamp()) {
      mCorrectState->update();
   }

   bool failed = false;

   failed |= verifyConnection(mConnection, mCorrectState, simTime);
   failed |= verifyLayer(mInputPublisher, mCorrectState->getCorrectInput(), simTime);
   failed |= verifyLayer(mOutputPublisher, mCorrectState->getCorrectOutput(), simTime);

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

bool MomentumConnViscosityCheckpointerTestProbe::verifyConnection(
      PV::ComponentBasedObject *connection,
      CorrectState const *correctState,
      double timevalue) {
   bool failed = false;

   if (mCommunicator->commRank() == 0 and correctState->doesWeightUpdate(timevalue)) {
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

bool MomentumConnViscosityCheckpointerTestProbe::verifyConnValue(
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

bool MomentumConnViscosityCheckpointerTestProbe::verifyLayer(
      PV::BasePublisherComponent *layer,
      float correctValue,
      double timevalue) {
   bool failed = false;

   PVLayerLoc const *inputLoc = layer->getLayerLoc();
   PVHalo const *inputHalo    = &inputLoc->halo;
   int const inputNxExt       = inputLoc->nx + inputHalo->lt + inputHalo->rt;
   int const inputNyExt       = inputLoc->ny + inputHalo->dn + inputHalo->up;
   PV::Buffer<float> localBuffer(layer->getLayerData(0), inputNxExt, inputNyExt, inputLoc->nf);
   PV::Communicator const *comm   = mCommunicator;
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
