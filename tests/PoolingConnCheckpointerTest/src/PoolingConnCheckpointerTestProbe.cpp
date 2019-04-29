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

using namespace PV;

PoolingConnCheckpointerTestProbe::PoolingConnCheckpointerTestProbe() {}

PoolingConnCheckpointerTestProbe::PoolingConnCheckpointerTestProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

PoolingConnCheckpointerTestProbe::~PoolingConnCheckpointerTestProbe() {}

void PoolingConnCheckpointerTestProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   return ColProbe::initialize(name, params, comm);
}

void PoolingConnCheckpointerTestProbe::ioParam_textOutputFlag(enum ParamsIOFlag ioFlag) {
   ColProbe::ioParam_textOutputFlag(ioFlag);
   if (ioFlag == PARAMS_IO_READ && !getTextOutputFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         ErrorLog()
               << getDescription()
               << ": PoolingConnCheckpointerTestProbe requires textOutputFlag to be set to true.\n";
      }
   }
}

Response::Status PoolingConnCheckpointerTestProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ColProbe::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   auto *objectTable = message->mObjectTable;

   status = mConnection ? status : status + initConnection(objectTable);
   status = mInputPublisher ? status : status + initInputPublisher(objectTable);
   status = mOutputPublisher ? status : status + initOutputPublisher(objectTable);
   if (!Response::completed(status)) {
      return status;
   }

   mInitializeFromCheckpointFlag = mInputPublisher->getInitializeFromCheckpointFlag();
   FatalIf(
         mInitializeFromCheckpointFlag != mOutputPublisher->getInitializeFromCheckpointFlag(),
         "%s and %s have different initializeFromCheckpointFlag values.\n",
         mInputPublisher->getDescription_c(),
         mOutputPublisher->getDescription_c());

   return Response::SUCCESS;
}

Response::Status
PoolingConnCheckpointerTestProbe::initConnection(ObserverTable const *objectTable) {
   char const *connectionName = "InputToOutput";
   auto *connection           = objectTable->findObject<PoolingConn>(connectionName);
   FatalIf(
         connection == nullptr, "column does not have a HyPerConn named \"%s\".\n", connectionName);
   if (checkCommunicatedFlag(connection) == Response::POSTPONE) {
      return Response::POSTPONE;
   }
   mConnection = connection;

   auto *patchSize = objectTable->findObject<PatchSize>(connectionName);
   FatalIf(
         patchSize == nullptr,
         "%s does not have a PatchSize component.\n",
         mConnection->getDescription_c());
   if (checkCommunicatedFlag(patchSize) == Response::POSTPONE) {
      return Response::POSTPONE;
   }
   FatalIf(patchSize->getPatchSizeX() != 1, "This test assumes that the connection has nxp==1.\n");
   FatalIf(patchSize->getPatchSizeY() != 1, "This test assumes that the connection has nyp==1.\n");
   FatalIf(patchSize->getPatchSizeF() != 1, "This test assumes that the connection has nfp==1.\n");
   return Response::SUCCESS;
}

Response::Status
PoolingConnCheckpointerTestProbe::initInputPublisher(ObserverTable const *objectTable) {
   char const *inputLayerName = "Input";
   auto *inputLayer           = objectTable->findObject<InputLayer>(std::string(inputLayerName));
   FatalIf(
         inputLayer == nullptr,
         "column does not have an InputLayer named \"%s\".\n",
         inputLayerName);
   if (checkCommunicatedFlag(inputLayer) == Response::POSTPONE) {
      return Response::POSTPONE;
   }

   PVHalo const *halo = &inputLayer->getLayerLoc()->halo;
   FatalIf(
         halo->lt != 0 || halo->rt != 0 || halo->dn != 0 || halo->up != 0,
         "This test assumes that the input layer has no border region.\n");

   auto *inputBuffer = objectTable->findObject<InputActivityBuffer>(inputLayerName);
   FatalIf(
         inputBuffer == nullptr,
         "%s does not have an InputActivityBuffer.\n",
         inputLayer->getDescription_c());
   if (checkCommunicatedFlag(inputBuffer) == Response::POSTPONE) {
      return Response::POSTPONE;
   }
   FatalIf(
         inputBuffer->getDisplayPeriod() != 4.0,
         "This test assumes that the display period is 4 (should really not be hard-coded.\n");

   mInputPublisher = objectTable->findObject<BasePublisherComponent>(inputLayerName);
   FatalIf(
         mInputPublisher == nullptr,
         "%s does not have a BasePublisherComponent.\n",
         inputLayer->getDescription_c());
   return Response::SUCCESS;
}

Response::Status
PoolingConnCheckpointerTestProbe::initOutputPublisher(ObserverTable const *objectTable) {
   char const *outputLayerName = "Output";
   auto *outputLayer           = objectTable->findObject<HyPerLayer>(outputLayerName);
   FatalIf(
         outputLayer == nullptr,
         "column does not have a HyPerLayer named \"%s\".\n",
         outputLayerName);
   if (checkCommunicatedFlag(outputLayer) == Response::POSTPONE) {
      return Response::POSTPONE;
   }

   mOutputPublisher = objectTable->findObject<BasePublisherComponent>(outputLayerName);
   FatalIf(
         mOutputPublisher == nullptr,
         "%s does not have a BasePublisherComponent.\n",
         outputLayer->getDescription_c());
   return Response::SUCCESS;
}

Response::Status
PoolingConnCheckpointerTestProbe::checkCommunicatedFlag(BaseObject *dependencyObject) {
   if (!dependencyObject->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->commRank() == 0) {
         InfoLog().printf(
               "%s must wait until \"%s\" has finished its communicateInitInfo stage.\n",
               getDescription_c(),
               dependencyObject->getName());
      }
      return Response::POSTPONE;
   }
   else {
      return Response::SUCCESS;
   }
}

Response::Status PoolingConnCheckpointerTestProbe::initializeState(
      std::shared_ptr<InitializeStateMessage const> message) {
   FatalIf(message->mDeltaTime != 1.0, "This test assumes that the HyPerCol dt is 1.0.\n");
   return Response::SUCCESS;
}

Response::Status
PoolingConnCheckpointerTestProbe::readStateFromCheckpoint(Checkpointer *checkpointer) {
   Checkpointer::TimeInfo timeInfo;
   CheckpointEntryData<Checkpointer::TimeInfo> timeInfoCheckpointEntry(
         std::string("timeinfo"),
         mCommunicator->getLocalMPIBlock(),
         &timeInfo,
         (size_t)1,
         true /*broadcast*/);
   std::string initializeFromCheckpointDir(checkpointer->getInitializeFromCheckpointDir());
   timeInfoCheckpointEntry.read(initializeFromCheckpointDir, nullptr);

   mStartingUpdateNumber = calcUpdateNumber(timeInfo.mSimTime);

   return Response::SUCCESS;
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

Response::Status PoolingConnCheckpointerTestProbe::outputState(double simTime, double deltaTime) {
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
   return Response::SUCCESS;
}

bool PoolingConnCheckpointerTestProbe::verifyLayer(
      BasePublisherComponent *layer,
      Buffer<float> const &correctValueBuffer,
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
   Communicator const *comm = mCommunicator;
   std::vector<int> badIndicesGlobal;
   if (comm->commRank() == 0) {
      badIndicesGlobal.resize(loc.nxGlobal * loc.nyGlobal * loc.nf);
      std::vector<MPI_Request> requests(comm->commSize() - 1);
      for (int r = 1; r < comm->commSize(); r++) {
         int *recvBuffer = &badIndicesGlobal.at(r * numNeurons);
         MPI_Irecv(recvBuffer, numNeurons, MPI_INT, r, 211, comm->communicator(), &requests[r - 1]);
      }
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
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
