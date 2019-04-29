/*
 * BaseConnectionProbe.cpp
 *
 *  Created on: Oct 20, 2011
 *      Author: pschultz
 */

#include "BaseConnectionProbe.hpp"

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

void BaseConnectionProbe::initOutputStreams(const char *filename, Checkpointer *checkpointer) {
   if (getMPIBlock()->getRank() == 0) {
      char const *probeOutputFilename = getProbeOutputFilename();
      if (probeOutputFilename) {
         std::string path(probeOutputFilename);
         std::ios_base::openmode mode = std::ios_base::out;
         if (!checkpointer->getCheckpointReadDirectory().empty()) {
            mode |= std::ios_base::app;
         }
         if (path[0] != '/') {
            path = checkpointer->makeOutputPathFilename(path);
         }
         auto stream = new FileStream(path.c_str(), mode, checkpointer->doesVerifyWrites());
         mOutputStreams.push_back(stream);
      }
      else {
         auto stream = new PrintStream(PV::getOutputStream());
         mOutputStreams.push_back(stream);
      }
   }
   else {
      mOutputStreams.clear();
   }
}

} // end of namespace PV
