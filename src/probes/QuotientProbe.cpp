#include "QuotientProbe.hpp"
#include "include/pv_common.h"
#include "observerpattern/BaseMessage.hpp"
#include "probes/ProbeData.hpp"
#include "utils/PVLog.hpp"
#include <functional>

namespace PV {

QuotientProbe::QuotientProbe(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

Response::Status QuotientProbe::allocateDataStructures() {
   Response::Status status = ProbeInterface::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   if (!mNumerator->getDataStructuresAllocatedFlag()) {
      InfoLog().printf(
            "%s must postpone until numerator %s allocates.\n",
            getDescription_c(),
            mNumerator->getDescription_c());
      status = status + Response::POSTPONE;
   }
   if (!mDenominator->getDataStructuresAllocatedFlag()) {
      InfoLog().printf(
            "%s must postpone until denominator %s allocates.\n",
            getDescription_c(),
            mDenominator->getDescription_c());
      status = status + Response::POSTPONE;
   }
   if (!Response::completed(status)) {
      return status;
   }
   int batchWidth = mNumerator->getNumValues();
   FatalIf(
         mDenominator->getNumValues() != batchWidth,
         "Numerator %s and denominator %s have different NumValues (%d versus %d)\n",
         mNumerator->getDescription_c(),
         mDenominator->getDescription_c(),
         batchWidth,
         mDenominator->getNumValues());
   setNumValues(batchWidth);
   return Response::SUCCESS;
}

void QuotientProbe::calcValues(double timestamp) {
   auto const &numerator   = mNumerator->getValues(timestamp);
   auto const &denominator = mDenominator->getValues(timestamp);
   ProbeData<double> quotient(timestamp, getNumValues(), 0.0);
   pvAssert(static_cast<int>(quotient.size()) == getNumValues());
   pvAssert(static_cast<int>(quotient.size()) == mNumerator->getNumValues());
   pvAssert(static_cast<int>(quotient.size()) == mDenominator->getNumValues());
   for (int b = 0; b < getNumValues(); ++b) {
      quotient.getValue(b) = numerator[b] / denominator[b];
   }
   mStoredValues.store(quotient);
   setValues(quotient);
}

Response::Status
QuotientProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ProbeInterface::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   auto *objectTable = message->mObjectTable;
   mNumerator        = objectTable->findObject<ProbeInterface>(mNumeratorName);
   mDenominator      = objectTable->findObject<ProbeInterface>(mDenominatorName);
   bool failed       = false;
   if (mNumerator == nullptr) {
      failed = true;
      ErrorLog().printf(
            "%s: numerator probe \"%s\" either does not exist or is not a suitable probe type.\n",
            getDescription_c(),
            mNumeratorName);
   }
   if (mDenominator == nullptr) {
      failed = true;
      ErrorLog().printf(
            "%s: denominator probe \"%s\" either does not exist or is not a suitable probe type.\n",
            getDescription_c(),
            mDenominatorName);
   }
   FatalIf(failed, "%s failed.\n", getDescription_c());
   return Response::SUCCESS;
}

void QuotientProbe::createComponents(char const *name, PVParams *params, Communicator const *comm) {
   // NB: the data members mName and mParams have not been set when createComponents() is called.
   createProbeOutputter(name, params, comm);
   createProbeTrigger(name, params);
}

void QuotientProbe::createProbeOutputter(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mProbeOutputter = std::make_shared<QuotientProbeOutputter>(name, params, comm);
}

void QuotientProbe::createProbeTrigger(char const *name, PVParams *params) {
   mProbeTrigger = std::make_shared<ProbeTriggerComponent>(name, params);
}

void QuotientProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   createComponents(name, params, comm);
   // createComponents() must be called before the base class's initialize(),
   // because BaseObject::initialize() calls the ioParamsFillGroup() method,
   // which calls each component's ioParamsFillGroup() method.
   ProbeInterface::initialize(name, params, comm);
}

void QuotientProbe::initMessageActionMap() {
   ProbeInterface::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ColProbeOutputStateMessage const>(msgptr);
      return respondColProbeOutputState(castMessage);
   };
   mMessageActionMap.emplace("ColProbeOutputState", action);
}

void QuotientProbe::ioParam_denominator(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, getName(), "numerator", &mNumeratorName);
}

void QuotientProbe::ioParam_numerator(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, getName(), "denominator", &mDenominatorName);
}

// QuotientProbe parameter valueDescription was marked obsolete Mar 6, 2023.
void QuotientProbe::ioParam_valueDescription(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      FatalIf(
            parameters()->stringPresent(getName(), "valueDescription"),
            "%s: valueDescription parameter is obsolete. Use the message parameter instead.\n",
            getDescription_c());
   }
}

int QuotientProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ProbeInterface::ioParamsFillGroup(ioFlag);
   mProbeOutputter->ioParamsFillGroup(ioFlag);
   mProbeTrigger->ioParamsFillGroup(ioFlag);
   ioParam_numerator(ioFlag);
   ioParam_denominator(ioFlag);
   ioParam_valueDescription(ioFlag);
   return status;
}

Response::Status QuotientProbe::outputState(double simTime, double deltaTime) {
   if (mProbeTrigger->needUpdate(simTime, deltaTime)) {
      getValues(simTime);
      // if needed, getValues() will call calcValues(), which will store the result
      // in mStoredValues
   }
   return Response::SUCCESS;
}

Response::Status QuotientProbe::prepareCheckpointWrite(double simTime) {
   mProbeOutputter->printBuffer(mStoredValues);
   mStoredValues.clear();
   return Response::SUCCESS;
}

Response::Status QuotientProbe::processCheckpointRead(double simTime) {
   // This assumes that if there is a trigger layer, the layer triggered
   // at the time of the checkpoint being read from; or at least that
   // the target layer has not updated since then.
   calcValues(simTime);

   // It's more convenient to call mStoredValues.store() from within calcValues().
   // However, writing the checkpoint clears the stored values; therefore
   // reading from checkpoint should conclude with an empty mStoredValues.
   mStoredValues.clear();

   return Response::SUCCESS;
}

Response::Status
QuotientProbe::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = ProbeInterface::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer = message->mDataRegistry;
   int localNBatch    = getNumValues();

   // Check that AllocateDataStructures stage completed; if so, the
   // QuotientProbe, Numerator, and Denominator should all have the same NumValues.
   FatalIf(
         !getDataStructuresAllocatedFlag(),
         "%s received RegisterData message before it had completed AllocateDataStructures stage\n",
         getDescription_c());
   localNBatch = getNumValues();
   pvAssert(mNumerator->getNumValues() == localNBatch);
   pvAssert(mDenominator->getNumValues() == localNBatch);

   mProbeOutputter->initOutputStreams(checkpointer, localNBatch);
   return Response::SUCCESS;
}

Response::Status QuotientProbe::respondColProbeOutputState(
      std::shared_ptr<ColProbeOutputStateMessage const>(message)) {
   return outputState(message->mTime, message->mDeltaTime);
}

} // namespace PV
