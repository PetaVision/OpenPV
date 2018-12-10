/*
 * PoolingConnCheckpointerTestProbe.cpp
 *
 *  Created on: Jan 5, 2017
 *      Author: pschultz
 */

#include "PoolingConnCheckpointerTestProbe.hpp"
#include "components/InputActivityBuffer.hpp"
#include "components/PatchSize.hpp"
#include "connections/PoolingConn.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include <algorithm>
#include <climits>
#include <cmath>

PoolingConnCheckpointerTestProbe::PoolingConnCheckpointerTestProbe() {}

PoolingConnCheckpointerTestProbe::PoolingConnCheckpointerTestProbe(
      const char *name,
      PV::PVParams *params,
      PV::Communicator *comm) {
   initialize(name, params, comm);
}

PoolingConnCheckpointerTestProbe::~PoolingConnCheckpointerTestProbe() {}

void PoolingConnCheckpointerTestProbe::initialize(
      const char *name,
      PV::PVParams *params,
      PV::Communicator *comm) {
   return PV::ColProbe::initialize(name, params, comm);
}

void PoolingConnCheckpointerTestProbe::ioParam_textOutputFlag(enum PV::ParamsIOFlag ioFlag) {
   ColProbe::ioParam_textOutputFlag(ioFlag);
   if (ioFlag == PV::PARAMS_IO_READ && !getTextOutputFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         ErrorLog()
               << getDescription()
               << ": PoolingConnCheckpointerTestProbe requires textOutputFlag to be set to true.\n";
      }
   }
}

PV::Response::Status PoolingConnCheckpointerTestProbe::communicateInitInfo(
      std::shared_ptr<PV::CommunicateInitInfoMessage const> message) {
   auto status = PV::ColProbe::communicateInitInfo(message);
   if (!PV::Response::completed(status)) {
      return status;
   }

   auto *componentTable = message->mHierarchy;

   status = mConnection ? status : status + initConnection(componentTable);
   status = mInputPublisher ? status : status + initInputPublisher(componentTable);
   status = mOutputPublisher ? status : status + initOutputPublisher(componentTable);
   if (!PV::Response::completed(status)) {
      return status;
   }

   mInitializeFromCheckpointFlag = mInputPublisher->getInitializeFromCheckpointFlag();
   FatalIf(
         mInitializeFromCheckpointFlag != mOutputPublisher->getInitializeFromCheckpointFlag(),
         "%s and %s have different initializeFromCheckpointFlag values.\n",
         mInputPublisher->getDescription_c(),
         mOutputPublisher->getDescription_c());

   return PV::Response::SUCCESS;
}

PV::Response::Status
PoolingConnCheckpointerTestProbe::initConnection(PV::ObserverTable const *componentTable) {
   auto *connection = componentTable->lookupByName<PV::PoolingConn>(std::string("InputToOutput"));
   FatalIf(connection == nullptr, "column does not have a HyPerConn named \"InputToOutput\".\n");
   if (checkCommunicatedFlag(connection) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }
   mConnection = connection;

   auto *patchSize = mConnection->getComponentByType<PV::PatchSize>();
   FatalIf(
         patchSize == nullptr,
         "%s does not have a PatchSize component.\n",
         mConnection->getDescription_c());
   FatalIf(patchSize->getPatchSizeX() != 1, "This test assumes that the connection has nxp==1.\n");
   FatalIf(patchSize->getPatchSizeY() != 1, "This test assumes that the connection has nyp==1.\n");
   FatalIf(patchSize->getPatchSizeF() != 1, "This test assumes that the connection has nfp==1.\n");
   return PV::Response::SUCCESS;
}

PV::Response::Status
PoolingConnCheckpointerTestProbe::initInputPublisher(PV::ObserverTable const *componentTable) {
   auto *inputLayer = componentTable->lookupByName<PV::InputLayer>(std::string("Input"));
   FatalIf(inputLayer == nullptr, "column does not have an InputLayer named \"Input\".\n");
   if (checkCommunicatedFlag(inputLayer) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }

   PVHalo const *halo = &inputLayer->getLayerLoc()->halo;
   FatalIf(
         halo->lt != 0 || halo->rt != 0 || halo->dn != 0 || halo->up != 0,
         "This test assumes that the input layer has no border region.\n");

   auto *activityComponent = inputLayer->getComponentByType<PV::ActivityComponent>();
   auto *inputBuffer       = activityComponent->getComponentByType<PV::InputActivityBuffer>();
   FatalIf(
         inputBuffer->getDisplayPeriod() != 4.0,
         "This test assumes that the display period is 4 (should really not be hard-coded.\n");

   mInputPublisher = inputLayer->getComponentByType<PV::PublisherComponent>();
   FatalIf(
         mInputPublisher == nullptr,
         "%s does not have a PublisherComponent.\n",
         inputLayer->getDescription_c());
   return PV::Response::SUCCESS;
}

PV::Response::Status
PoolingConnCheckpointerTestProbe::initOutputPublisher(PV::ObserverTable const *componentTable) {
   auto *outputLayer = componentTable->lookupByName<PV::HyPerLayer>(std::string("Output"));
   FatalIf(outputLayer == nullptr, "column does not have a HyPerLayer named \"Output\".\n");
   if (checkCommunicatedFlag(outputLayer) == PV::Response::POSTPONE) {
      return PV::Response::POSTPONE;
   }

   mOutputPublisher = outputLayer->getComponentByType<PV::PublisherComponent>();
   FatalIf(
         mOutputPublisher == nullptr,
         "%s does not have a PublisherComponent.\n",
         outputLayer->getDescription_c());
   return PV::Response::SUCCESS;
}

PV::Response::Status
PoolingConnCheckpointerTestProbe::checkCommunicatedFlag(PV::BaseObject *dependencyObject) {
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

PV::Response::Status PoolingConnCheckpointerTestProbe::initializeState(
      std::shared_ptr<PV::InitializeStateMessage const> message) {
   FatalIf(message->mDeltaTime != 1.0, "This test assumes that the HyPerCol dt is 1.0.\n");
   return PV::Response::SUCCESS;
}

PV::Response::Status
PoolingConnCheckpointerTestProbe::readStateFromCheckpoint(PV::Checkpointer *checkpointer) {
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

int PoolingConnCheckpointerTestProbe::calcUpdateNumber(double timevalue) {
   pvAssert(timevalue >= 0.0);
   int const step = (int)std::nearbyint(timevalue);
   pvAssert(step >= 0);
   int const updateNumber = (step + 3) / 4; // integer division
   return updateNumber;
}

void PoolingConnCheckpointerTestProbe::initializeCorrectValues(double timevalue) {
   int const updateNumber = mStartingUpdateNumber + calcUpdateNumber(timevalue);
   mCorrectState          = new CorrectState(
         updateNumber - 1, mInputPublisher->getLayerLoc(), mOutputPublisher->getLayerLoc());
   // Don't update for the current updateNumber;
   // outputState calls mCorrectState->update() if needed.
}

PV::Response::Status
PoolingConnCheckpointerTestProbe::outputState(double simTime, double deltaTime) {
   if (!mValuesSet) {
      initializeCorrectValues(simTime);
      mValuesSet = true;
   }
   int const updateNumber = mStartingUpdateNumber + calcUpdateNumber(simTime);
   while (updateNumber > mCorrectState->getUpdateNumber()) {
      mCorrectState->update();
   }

   bool failed = false;

   failed |= verifyLayer(mInputPublisher, mCorrectState->getCorrectInputBuffer(), simTime);
   failed |= verifyLayer(mOutputPublisher, mCorrectState->getCorrectOutputBuffer(), simTime);

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

bool PoolingConnCheckpointerTestProbe::verifyLayer(
      PV::PublisherComponent *layer,
      PV::Buffer<float> const &correctValueBuffer,
      double timevalue) {
   int failed = 0;

   float const *layerData = layer->getLayerData();
   PVLayerLoc loc         = *layer->getLayerLoc();
   int const nx           = loc.nx;
   int const ny           = loc.ny;
   int const nf           = loc.nf;
   int const numNeurons   = nx * ny * nf;
   std::vector<int> badIndices(numNeurons, -1);
   for (int k = 0; k < numNeurons; k++) {
      int const x = kxPos(k, nx, ny, nf);
      int const y = kyPos(k, nx, ny, nf);
      int const f = featureIndex(k, nx, ny, nf);
      if (layerData[k] != correctValueBuffer.at(x, y, f)) {
         int const kGlobal = globalIndexFromLocal(k, loc);
         badIndices[k]     = kGlobal;
         failed            = 1;
      }
   }
   PV::Communicator *comm = mCommunicator;
   std::vector<int> badIndicesGlobal;
   if (comm->commRank() == 0) {
      badIndicesGlobal.resize(loc.nxGlobal * loc.nyGlobal * loc.nf);
      std::vector<MPI_Request> requests(comm->commSize() - 1);
      for (int r = 1; r < comm->commSize(); r++) {
         int *recvBuffer = &badIndicesGlobal.at(r * numNeurons);
         MPI_Irecv(recvBuffer, numNeurons, MPI_INT, r, 211, comm->communicator(), &requests[r - 1]);
      }
      int status = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
      badIndicesGlobal.erase(
            std::remove_if(
                  badIndicesGlobal.begin(), badIndicesGlobal.end(), [](int j) { return j < 0; }),
            badIndicesGlobal.end());
      std::sort(badIndicesGlobal.begin(), badIndicesGlobal.end());
   }
   else {
      MPI_Send(badIndices.data(), numNeurons, MPI_INT, 0, 211, comm->communicator());
   }

   MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_LOR, comm->communicator());
   return failed != 0;
}
