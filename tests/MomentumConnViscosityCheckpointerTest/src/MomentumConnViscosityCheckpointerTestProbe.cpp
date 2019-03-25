/*
 * MomentumConnViscosityCheckpointerTestProbe.cpp
 *
 *  Created on: Jan 6, 2017
 *      Author: pschultz
 */

#include "MomentumConnViscosityCheckpointerTestProbe.hpp"
#include <cmath>
#include <utils/BufferUtilsMPI.hpp>

MomentumConnViscosityCheckpointerTestProbe::MomentumConnViscosityCheckpointerTestProbe() {
   initialize_base();
}

MomentumConnViscosityCheckpointerTestProbe::MomentumConnViscosityCheckpointerTestProbe(
      const char *name,
      PV::HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

MomentumConnViscosityCheckpointerTestProbe::~MomentumConnViscosityCheckpointerTestProbe() {}

int MomentumConnViscosityCheckpointerTestProbe::initialize_base() { return PV_SUCCESS; }

int MomentumConnViscosityCheckpointerTestProbe::initialize(const char *name, PV::HyPerCol *hc) {
   int status = PV::ColProbe::initialize(name, hc);
   FatalIf(parent->getDeltaTime() != 1.0, "This test assumes that the HyPerCol dt is 1.0.\n");
   return status;
}

void MomentumConnViscosityCheckpointerTestProbe::ioParam_textOutputFlag(
      enum PV::ParamsIOFlag ioFlag) {
   ColProbe::ioParam_textOutputFlag(ioFlag);
   if (ioFlag == PV::PARAMS_IO_READ && !getTextOutputFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
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

   status = status + initInputLayer(message);
   status = status + initOutputLayer(message);
   status = status + initConnection(message);
   return status;
}

PV::Response::Status MomentumConnViscosityCheckpointerTestProbe::initInputLayer(
      std::shared_ptr<PV::CommunicateInitInfoMessage const> message) {
   mInputLayer = message->lookup<PV::InputLayer>(std::string("Input"));
   FatalIf(mInputLayer == nullptr, "column does not have an InputLayer named \"Input\".\n");
   if (checkCommunicatedFlag(mInputLayer) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }

   FatalIf(
         mInputLayer->getDisplayPeriod() != 4.0,
         "This test assumes that the display period is 4 (should really not be hard-coded.\n");
   return PV::Response::SUCCESS;
}

PV::Response::Status MomentumConnViscosityCheckpointerTestProbe::initOutputLayer(
      std::shared_ptr<PV::CommunicateInitInfoMessage const> message) {
   mOutputLayer = message->lookup<PV::HyPerLayer>(std::string("Output"));
   FatalIf(mOutputLayer == nullptr, "column does not have a HyPerLayer named \"Output\".\n");
   if (checkCommunicatedFlag(mOutputLayer) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }
   return PV::Response::SUCCESS;
}

PV::Response::Status MomentumConnViscosityCheckpointerTestProbe::initConnection(
      std::shared_ptr<PV::CommunicateInitInfoMessage const> message) {
   mConnection = message->lookup<PV::MomentumConn>(std::string("InputToOutput"));
   FatalIf(
         mConnection == nullptr, "column does not have a MomentumConn named \"InputToOutput\".\n");
   if (checkCommunicatedFlag(mConnection) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }

   FatalIf(
         mConnection->getNumAxonalArbors() != 1,
         "This test assumes that the connection has only 1 arbor.\n");
   FatalIf(
         mConnection->getDelay(0) != 0.0,
         "This test assumes that the connection has zero delay.\n");
   FatalIf(
         !mConnection->getSharedWeights(),
         "This test assumes that the connection is using shared weights.\n");
   FatalIf(
         mConnection->getPatchSizeX() != 1, "This test assumes that the connection has nxp==1.\n");
   FatalIf(
         mConnection->getPatchSizeY() != 1, "This test assumes that the connection has nyp==1.\n");
   FatalIf(
         mConnection->getPatchSizeF() != 1, "This test assumes that the connection has nfp==1.\n");
   FatalIf(
         std::strcmp(mConnection->getMomentumMethod(), "viscosity"),
         "This test assumes that the connection has momentumMethod=\"viscosity\".\n");
   return PV::Response::SUCCESS;
}

PV::Response::Status MomentumConnViscosityCheckpointerTestProbe::checkCommunicatedFlag(
      PV::BaseObject *dependencyObject) {
   if (!dependencyObject->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->commRank() == 0) {
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

PV::Response::Status MomentumConnViscosityCheckpointerTestProbe::readStateFromCheckpoint(
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

   mStartingTimestamp = timeInfo.mSimTime;

   return PV::Response::SUCCESS;
}

void MomentumConnViscosityCheckpointerTestProbe::initializeCorrectValues(double timevalue) {
   int const updateNumber = mStartingTimestamp + timevalue;
   float const tau        = mConnection->getTimeConstantTau();
   float const tau_exp    = std::exp(-1.0f / tau);
   mCorrectState =
         new CorrectState(tau_exp, 1.0f /*weight*/, 0.0f /*dw*/, 1.0f /*input*/, 1.0f /*output*/);
}

PV::Response::Status MomentumConnViscosityCheckpointerTestProbe::outputState(double timevalue) {
   if (!mValuesSet) {
      initializeCorrectValues(timevalue);
      mValuesSet = true;
   }
   int const updateNumber = mStartingTimestamp + timevalue;
   while (updateNumber > mCorrectState->getTimestamp()) {
      mCorrectState->update();
   }

   bool failed = false;

   failed |= verifyConnection(mConnection, mCorrectState, timevalue);
   failed |= verifyLayer(mInputLayer, mCorrectState->getCorrectInput(), timevalue);
   failed |= verifyLayer(mOutputLayer, mCorrectState->getCorrectOutput(), timevalue);
   InfoLog().printf(
         "t=%f, input=%f, output=%f, W=%f, dW=%f\n",
         timevalue,
         (double)mInputLayer->getLayerData(0)[0],
         (double)mOutputLayer->getLayerData(0)[0],
         (double)mConnection->getWeightsDataHead(0, 0)[0],
         (double)mConnection->getDeltaWeightsDataHead(0, 0)[0]);

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
   // Test runs all timesteps and then checks the mTestFailed flag at the end.
   return PV::Response::SUCCESS;
}

bool MomentumConnViscosityCheckpointerTestProbe::verifyConnection(
      PV::MomentumConn *connection,
      CorrectState const *correctState,
      double timevalue) {
   bool failed = false;

   if (parent->getCommunicator()->commRank() == 0 and correctState->doesWeightUpdate(timevalue)) {
      float observedWeightValue = connection->getWeightsDataStart(0)[0];
      float correctWeightValue  = correctState->getCorrectWeight();
      failed |= verifyConnValue(timevalue, observedWeightValue, correctWeightValue, "weight");

      float observed_dwValue = connection->getDeltaWeightsDataStart(0)[0];
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

bool MomentumConnViscosityCheckpointerTestProbe::verifyLayer(
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
