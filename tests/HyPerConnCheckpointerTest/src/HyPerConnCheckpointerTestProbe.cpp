/*
 * HyPerConnCheckpointerTestProbe.cpp
 *
 *  Created on: Jan 5, 2017
 *      Author: pschultz
 */

#include "HyPerConnCheckpointerTestProbe.hpp"
#include <cmath>
#include <utils/BufferUtilsMPI.hpp>

HyPerConnCheckpointerTestProbe::HyPerConnCheckpointerTestProbe() { initialize_base(); }

HyPerConnCheckpointerTestProbe::HyPerConnCheckpointerTestProbe(const char *name, PV::HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

HyPerConnCheckpointerTestProbe::~HyPerConnCheckpointerTestProbe() {}

int HyPerConnCheckpointerTestProbe::initialize_base() { return PV_SUCCESS; }

int HyPerConnCheckpointerTestProbe::initialize(const char *name, PV::HyPerCol *hc) {
   int status = PV::ColProbe::initialize(name, hc);
   FatalIf(parent->getDeltaTime() != 1.0, "This test assumes that the HyPerCol dt is 1.0.\n");
   return status;
}

void HyPerConnCheckpointerTestProbe::ioParam_textOutputFlag(enum PV::ParamsIOFlag ioFlag) {
   ColProbe::ioParam_textOutputFlag(ioFlag);
   if (ioFlag == PV::PARAMS_IO_READ && !getTextOutputFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog()
               << getDescription()
               << ": HyPerConnCheckpointerTestProbe requires textOutputFlag to be set to true.\n";
      }
   }
}

PV::Response::Status HyPerConnCheckpointerTestProbe::communicateInitInfo(
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

PV::Response::Status HyPerConnCheckpointerTestProbe::initInputLayer(
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

PV::Response::Status HyPerConnCheckpointerTestProbe::initOutputLayer(
      std::shared_ptr<PV::CommunicateInitInfoMessage const> message) {
   mOutputLayer = message->lookup<PV::HyPerLayer>(std::string("Output"));
   FatalIf(mOutputLayer == nullptr, "column does not have a HyPerLayer named \"Output\".\n");
   if (checkCommunicatedFlag(mOutputLayer) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }
   return PV::Response::SUCCESS;
}

PV::Response::Status HyPerConnCheckpointerTestProbe::initConnection(
      std::shared_ptr<PV::CommunicateInitInfoMessage const> message) {
   mConnection = message->lookup<PV::HyPerConn>(std::string("InputToOutput"));
   FatalIf(mConnection == nullptr, "column does not have a HyPerConn named \"InputToOutput\".\n");
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
   return PV::Response::SUCCESS;
}

PV::Response::Status
HyPerConnCheckpointerTestProbe::checkCommunicatedFlag(PV::BaseObject *dependencyObject) {
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

PV::Response::Status
HyPerConnCheckpointerTestProbe::readStateFromCheckpoint(PV::Checkpointer *checkpointer) {
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

   return PV::Response::SUCCESS;
}

int HyPerConnCheckpointerTestProbe::calcUpdateNumber(double timevalue) {
   pvAssert(timevalue >= 0.0);
   int const step = (int)std::nearbyint(timevalue);
   pvAssert(step >= 0);
   int const updateNumber = (step + 3) / 4; // integer division
   return updateNumber;
}

void HyPerConnCheckpointerTestProbe::initializeCorrectValues(double timevalue) {
   int const updateNumber = mStartingUpdateNumber + calcUpdateNumber(timevalue);
   if (updateNumber == 0) {
      mCorrectState = new CorrectState(0, 1.0f /*weight*/, 1.0f /*input*/, 2.0f /*output*/);
   }
   else {
      mCorrectState = new CorrectState(1, 3.0f /*weight*/, 1.0f /*input*/, 3.0f /*output*/);

      for (int j = 2; j < updateNumber; j++) {
         mCorrectState->update();
      }
      // Don't update for the current updateNumber; this will happen in outputState.
   }
}

PV::Response::Status HyPerConnCheckpointerTestProbe::outputState(double timestamp) {
   if (!mValuesSet) {
      initializeCorrectValues(timestamp);
      mValuesSet = true;
   }
   int const updateNumber = mStartingUpdateNumber + calcUpdateNumber(timestamp);
   while (updateNumber > mCorrectState->getUpdateNumber()) {
      mCorrectState->update();
   }

   bool failed = false;

   failed |= verifyConnection(mConnection, mCorrectState->getCorrectWeight(), timestamp);
   failed |= verifyLayer(mInputLayer, mCorrectState->getCorrectInput(), timestamp);
   failed |= verifyLayer(mOutputLayer, mCorrectState->getCorrectOutput(), timestamp);

   if (failed) {
      std::string errorMsg(getDescription() + " failed at t = " + std::to_string(timestamp) + "\n");
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
               "%s found all correct values at time %f\n", getDescription_c(), timestamp);
      }
   }
   // The test runs all timesteps and then checks the mTestFailed flag at the end.
   return PV::Response::SUCCESS;
}

bool HyPerConnCheckpointerTestProbe::verifyLayer(
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

bool HyPerConnCheckpointerTestProbe::verifyConnection(
      PV::HyPerConn *connection,
      float correctValue,
      double timevalue) {
   bool failed = false;

   if (parent->getCommunicator()->commRank() == 0) {
      FatalIf(
            mOutputStreams.empty(),
            "%s has empty mOutputStreams in root process.\n",
            getDescription_c());
      float observedWeightValue = connection->getWeightsDataStart(0)[0];
      if (observedWeightValue != correctValue) {
         output(0).printf(
               "Time %f, weight is %f, instead of the expected %f.\n",
               timevalue,
               (double)observedWeightValue,
               (double)correctValue);
         failed = true;
      }
   }
   return failed;
}
