/*
 * ColProbe.cpp
 *
 *  Created on: Nov 25, 2010
 *      Author: pschultz
 */

#include "ColProbe.hpp"

namespace PV {

ColProbe::ColProbe() { // Default constructor to be called by derived classes.
   // They should call ColProbe::initialize from their own initialization routine
   // instead of calling a non-default constructor.
   initialize_base();
}

ColProbe::ColProbe(const char *name, PVParams *params, Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
}

ColProbe::~ColProbe() {}

int ColProbe::initialize_base() { return PV_SUCCESS; }

void ColProbe::initialize(const char *name, PVParams *params, Communicator *comm) {
   BaseProbe::initialize(name, params, comm);
}

void ColProbe::initMessageActionMap() {
   BaseProbe::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ColProbeWriteParamsMessage const>(msgptr);
      return respondColProbeWriteParams(castMessage);
   };
   mMessageActionMap.emplace("ColProbeWriteParams", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ColProbeOutputStateMessage const>(msgptr);
      return respondColProbeOutputState(castMessage);
   };
   mMessageActionMap.emplace("ColProbeOutputState", action);
}

int ColProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = PV::BaseProbe::ioParamsFillGroup(ioFlag);
   return status;
}

void ColProbe::ioParam_targetName(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      targetName = strdup("");
   }
}

void ColProbe::initOutputStreams(const char *filename, Checkpointer *checkpointer) {
   BaseProbe::initOutputStreams(filename, checkpointer);
   outputHeader();
}

Response::Status
ColProbe::respondColProbeWriteParams(std::shared_ptr<ColProbeWriteParamsMessage const>(message)) {
   writeParams();
   return Response::SUCCESS;
}

Response::Status
ColProbe::respondColProbeOutputState(std::shared_ptr<ColProbeOutputStateMessage const>(message)) {
   return outputStateWrapper(message->mTime, message->mDeltaTime);
}

Response::Status
ColProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return BaseProbe::communicateInitInfo(message);
}

} // end namespace PV
