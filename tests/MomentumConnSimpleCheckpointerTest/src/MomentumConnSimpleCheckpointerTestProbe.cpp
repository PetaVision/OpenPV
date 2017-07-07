/*
 * MomentumConnSimpleCheckpointerTestProbe.cpp
 *
 *  Created on: Jan 6, 2017
 *      Author: pschultz
 */

#include "MomentumConnSimpleCheckpointerTestProbe.hpp"
#include <cmath>
#include <utils/BufferUtilsMPI.hpp>

MomentumConnSimpleCheckpointerTestProbe::MomentumConnSimpleCheckpointerTestProbe() {
   initialize_base();
}

MomentumConnSimpleCheckpointerTestProbe::MomentumConnSimpleCheckpointerTestProbe(
      const char *name,
      PV::HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

MomentumConnSimpleCheckpointerTestProbe::~MomentumConnSimpleCheckpointerTestProbe() {}

int MomentumConnSimpleCheckpointerTestProbe::initialize_base() { return PV_SUCCESS; }

int MomentumConnSimpleCheckpointerTestProbe::initialize(const char *name, PV::HyPerCol *hc) {
   int status = PV::ColProbe::initialize(name, hc);
   FatalIf(parent->getDeltaTime() != 1.0, "This test assumes that the HyPerCol dt is 1.0.\n");
   return status;
}

void MomentumConnSimpleCheckpointerTestProbe::ioParam_textOutputFlag(enum PV::ParamsIOFlag ioFlag) {
   ColProbe::ioParam_textOutputFlag(ioFlag);
   if (ioFlag == PV::PARAMS_IO_READ && !getTextOutputFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog() << getDescription() << ": MomentumConnSimpleCheckpointerTestProbe requires "
                                           "textOutputFlag to be set to true.\n";
      }
   }
}

int MomentumConnSimpleCheckpointerTestProbe::communicateInitInfo(
      std::shared_ptr<PV::CommunicateInitInfoMessage const> message) {
   int status = PV::ColProbe::communicateInitInfo(message);
   FatalIf(
         status != PV_SUCCESS, "%s failed in ColProbe::communicateInitInfo\n", getDescription_c());

   if (initInputLayer(message) == PV_POSTPONE) {
      return PV_POSTPONE;
   }
   if (initOutputLayer(message) == PV_POSTPONE) {
      return PV_POSTPONE;
   }
   if (initConnection(message) == PV_POSTPONE) {
      return PV_POSTPONE;
   }
   return status;
}

int MomentumConnSimpleCheckpointerTestProbe::initInputLayer(
      std::shared_ptr<PV::CommunicateInitInfoMessage const> message) {
   mInputLayer = message->lookup<PV::InputLayer>(std::string("Input"));
   FatalIf(mInputLayer == nullptr, "column does not have an InputLayer named \"Input\".\n");
   if (checkCommunicatedFlag(mInputLayer) == PV_POSTPONE) {
      return PV_POSTPONE;
   }

   FatalIf(
         mInputLayer->getDisplayPeriod() != 4.0,
         "This test assumes that the display period is 4 (should really not be hard-coded.\n");
   return PV_SUCCESS;
}

int MomentumConnSimpleCheckpointerTestProbe::initOutputLayer(
      std::shared_ptr<PV::CommunicateInitInfoMessage const> message) {
   mOutputLayer = message->lookup<PV::HyPerLayer>(std::string("Output"));
   FatalIf(mOutputLayer == nullptr, "column does not have a HyPerLayer named \"Output\".\n");
   if (checkCommunicatedFlag(mOutputLayer) == PV_POSTPONE) {
      return PV_POSTPONE;
   }
   return PV_SUCCESS;
}

int MomentumConnSimpleCheckpointerTestProbe::initConnection(
      std::shared_ptr<PV::CommunicateInitInfoMessage const> message) {
   mConnection = message->lookup<PV::MomentumConn>(std::string("InputToOutput"));
   FatalIf(
         mConnection == nullptr, "column does not have a MomentumConn named \"InputToOutput\".\n");
   if (checkCommunicatedFlag(mConnection) == PV_POSTPONE) {
      return PV_POSTPONE;
   }

   FatalIf(
         mConnection->numberOfAxonalArborLists() != 1,
         "This test assumes that the connection has only 1 arbor.\n");
   FatalIf(
         mConnection->getDelay(0) != 0.0,
         "This test assumes that the connection has zero delay.\n");
   FatalIf(
         !mConnection->usingSharedWeights(),
         "This test assumes that the connection is using shared weights.\n");
   FatalIf(mConnection->xPatchSize() != 1, "This test assumes that the connection has nxp==1.\n");
   FatalIf(mConnection->yPatchSize() != 1, "This test assumes that the connection has nyp==1.\n");
   FatalIf(mConnection->fPatchSize() != 1, "This test assumes that the connection has nfp==1.\n");
   FatalIf(
         std::strcmp(mConnection->getMomentumMethod(), "simple"),
         "This test assumes that the connection has momentumMethod=\"simple\".\n");
   return PV_SUCCESS;
}

int MomentumConnSimpleCheckpointerTestProbe::checkCommunicatedFlag(
      PV::BaseObject *dependencyObject) {
   if (!dependencyObject->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->commRank() == 0) {
         InfoLog().printf(
               "%s must wait until \"%s\" has finished its communicateInitInfo stage.\n",
               getDescription_c(),
               dependencyObject->getName());
      }
      return PV_POSTPONE;
   }
   else {
      return PV_SUCCESS;
   }
}

int MomentumConnSimpleCheckpointerTestProbe::readStateFromCheckpoint(
      PV::Checkpointer *checkpointer) {
   PV::Checkpointer::TimeInfo timeInfo;
   PV::CheckpointEntryData<PV::Checkpointer::TimeInfo> timeInfoCheckpointEntry(
         std::string("timeinfo"),
         parent->getCommunicator()->getLocalMPIBlock(),
         &timeInfo,
         (size_t)1,
         true /*broadcast*/);
   std::string initializeFromCheckpointDir(checkpointer->getInitializeFromCheckpointDir());
   timeInfoCheckpointEntry.read(initializeFromCheckpointDir, nullptr);

   mStartingUpdateNumber = calcUpdateNumber(timeInfo.mSimTime);

   return PV_SUCCESS;
}

int MomentumConnSimpleCheckpointerTestProbe::calcUpdateNumber(double timevalue) {
   pvAssert(timevalue >= parent->getStartTime());
   int const step = (int)std::nearbyint(timevalue - parent->getStartTime());
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

int MomentumConnSimpleCheckpointerTestProbe::outputState(double timevalue) {
   if (!mValuesSet) {
      initializeCorrectValues(timevalue);
      mValuesSet = true;
   }
   int const updateNumber = mStartingUpdateNumber + calcUpdateNumber(timevalue);
   while (updateNumber > mCorrectState->getUpdateNumber()) {
      mCorrectState->update();
   }

   bool failed = false;

   failed |= verifyConnection(mConnection, mCorrectState, timevalue);
   failed |= verifyLayer(mInputLayer, mCorrectState->getCorrectInput(), timevalue);
   failed |= verifyLayer(mOutputLayer, mCorrectState->getCorrectOutput(), timevalue);

   if (failed) {
      std::string errorMsg(getDescription() + " failed at t = " + std::to_string(timevalue) + "\n");
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
         output(0).printf(
               "%s found all correct values at time %f\n", getDescription_c(), timevalue);
      }
   }
   return PV_SUCCESS; // Test runs all timesteps and then checks the mTestFailed flag at the end.
}

bool MomentumConnSimpleCheckpointerTestProbe::verifyConnection(
      PV::MomentumConn *connection,
      CorrectState const *correctState,
      double timevalue) {
   bool failed = false;

   if (parent->getCommunicator()->commRank() == 0) {
      float observedWeightValue = connection->get_wDataStart(0)[0];
      float correctWeightValue  = correctState->getCorrectWeight();
      failed |= verifyConnValue(timevalue, observedWeightValue, correctWeightValue, "weight");

      float observed_dwValue = connection->get_dwDataStart(0)[0];
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
         parent->getCommunicator()->commRank() != 0,
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
   PV::Communicator *comm         = parent->getCommunicator();
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
