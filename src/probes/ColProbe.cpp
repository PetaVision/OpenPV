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

ColProbe::ColProbe(const char *name, PVParams *params, Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

ColProbe::~ColProbe() {}

int ColProbe::initialize_base() { return PV_SUCCESS; }

void ColProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
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

void ColProbe::initOutputStreams(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   BaseProbe::initOutputStreams(message);
   auto *checkpointer = message->mDataRegistry;
   outputHeader(checkpointer);
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

Response::Status ColProbe::outputStateStats(double simTime, double deltaTime) {
   getValues(simTime);
   auto &valuesVector = getProbeValues();
   int nbatch           = getNumValues();
   pvAssert(static_cast<int>(valuesVector.size()) == nbatch);
   double min = std::numeric_limits<double>::infinity();
   double max = -std::numeric_limits<double>::infinity();
   double sum = 0.0;
   for (int k=0; k < getNumValues(); k++) {
      double v = valuesVector[k];
      min = min < v ? min : valuesVector[k];
      max = max > v ? max : valuesVector[k];
      sum += v;
   }
   MPI_Comm const batchComm = mCommunicator->batchCommunicator();
   MPI_Allreduce(MPI_IN_PLACE, &min, 1, MPI_DOUBLE, MPI_MIN, batchComm);
   MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_DOUBLE, MPI_MAX, batchComm);
   MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, batchComm);
   if (!mOutputStreams.empty()) {
      pvAssert(mCommunicator->globalCommRank() == 0);
      pvAssert(mOutputStreams.size() == (std::size_t)1);
      double mean = sum/(getNumValues() * mCommunicator->numCommBatches());
      if (!isWritingToFile()) {
         output(0) << "\"" << name << "\","; // lack of \n is deliberate
      }
      output(0).printf("t=%10f, min=%10.9f, max=%10.9f, mean=%10.9f\n", simTime, min, max, mean);
   }
   else {
      pvAssert(mCommunicator->globalCommRank() != 0);
   }
   return Response::SUCCESS;
}

} // end namespace PV
