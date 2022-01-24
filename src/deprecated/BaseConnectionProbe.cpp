/*
 * BaseConnectionProbe.cpp
 *
 *  Created on: Oct 20, 2011
 *      Author: pschultz
 */

#include "BaseConnectionProbe.hpp"
#include "checkpointing/CheckpointEntryFilePosition.hpp"
#include "io/FileStreamBuilder.hpp"

namespace PV {

BaseConnectionProbe::BaseConnectionProbe() {}

BaseConnectionProbe::BaseConnectionProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

BaseConnectionProbe::~BaseConnectionProbe() { delete mIOTimer; }

void BaseConnectionProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   BaseProbe::initialize(name, params, comm);
}

void BaseConnectionProbe::initMessageActionMap() {
   BaseProbe::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ConnectionProbeWriteParamsMessage const>(msgptr);
      return respondConnectionProbeWriteParams(castMessage);
   };
   mMessageActionMap.emplace("ConnectionProbeWriteParams", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ConnectionOutputMessage const>(msgptr);
      return respondConnectionOutput(castMessage);
   };
   mMessageActionMap.emplace("ConnectionOutput", action);
}

void BaseConnectionProbe::ioParam_targetName(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(ioFlag, name, "targetConnection", &targetName, NULL, false);
   if (targetName == NULL) {
      BaseProbe::ioParam_targetName(ioFlag);
   }
}

Response::Status BaseConnectionProbe::respondConnectionProbeWriteParams(
      std::shared_ptr<ConnectionProbeWriteParamsMessage const> message) {
   writeParams();
   return Response::SUCCESS;
}

Response::Status BaseConnectionProbe::respondConnectionOutput(
      std::shared_ptr<ConnectionOutputMessage const> message) {
   mIOTimer->start();
   Response::Status status = outputStateWrapper(message->mTime, message->mDeltaT);
   mIOTimer->stop();
   return status;
}

Response::Status BaseConnectionProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseProbe::communicateInitInfo(message);
   if (status != Response::SUCCESS) {
      return status;
   }

   bool failed = false;
   mTargetConn = message->mObjectTable->findObject<ComponentBasedObject>(targetName);
   FatalIf(
         mTargetConn == nullptr,
         "%s, rank %d process: targetConnection \"%s\" is not a connection in the column.\n",
         getDescription_c(),
         mCommunicator->globalCommRank(),
         targetName);
   return Response::SUCCESS;
}

Response::Status BaseConnectionProbe::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = BaseProbe::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   mIOTimer = new Timer(getName(), "probe", "io     ");
   message->mDataRegistry->registerTimer(mIOTimer);
   return Response::SUCCESS;
}

void BaseConnectionProbe::initOutputStreams(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   if (getCommunicator()->getIOMPIBlock()->getRank() == 0) {
      auto *checkpointer = message->mDataRegistry;
      char const *probeOutputFilename = getProbeOutputFilename();
      if (probeOutputFilename and probeOutputFilename[0]) {
         std::string path(probeOutputFilename);
         auto fileManager = getCommunicator()->getOutputFileManager();
         bool createFlag = checkpointer->getCheckpointReadDirectory().empty();
         auto stream = FileStreamBuilder(
               fileManager,
               path,
               true /*textFile*/,
               false /*readOnlyFlag*/,
               createFlag,
               checkpointer->doesVerifyWrites()).get();
         auto checkpointEntry = std::make_shared<CheckpointEntryFilePosition>(
               getProbeOutputFilename(), stream);
         bool registerSucceeded = checkpointer->registerCheckpointEntry(
               checkpointEntry, false /*not constant for entire run*/);
         FatalIf(
               !registerSucceeded,
               "%s failed to register %s for checkpointing.\n",
               getDescription_c(),
               checkpointEntry->getName().c_str());
         mOutputStreams.push_back(stream);
      }
      else {
         auto stream = std::make_shared<PrintStream>(PV::getOutputStream());
         mOutputStreams.push_back(stream);
      }
   }
   else {
      mOutputStreams.clear();
   }
}

} // end of namespace PV
