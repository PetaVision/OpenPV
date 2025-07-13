#include "ColumnEnergyProbe.hpp"
#include "include/pv_common.h"
#include "observerpattern/BaseMessage.hpp"
#include "probes/ProbeData.hpp"
#include "utils/PVLog.hpp"
#include <functional>

namespace PV {

ColumnEnergyProbe::ColumnEnergyProbe(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

Response::Status ColumnEnergyProbe::allocateDataStructures() {
   Response::Status status = ProbeInterface::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   FatalIf(
         mTerms.empty(), "%s has no probes attached to it.\n", getDescription_c());
   for (auto *term : mTerms) {
      if (!term->getDataStructuresAllocatedFlag()) {
         InfoLog().printf(
               "%s must postpone until %s allocates.\n",
               getDescription_c(),
               term->getDescription_c());
         status = status + Response::POSTPONE;
      }
   }
   if (!Response::completed(status)) {
      return status;
   }
   int batchWidth = 0;
   bool diffSizes = false;
   if (!mTerms.empty()) {
      batchWidth = mTerms[0]->getNumValues();
      for (auto k = mTerms.begin(); k != mTerms.end(); ++k) {
         auto const &term = *k;
         if (term->getNumValues() != batchWidth) {
            ErrorLog().printf(
                  "%s: probes %s and %s have differing NumValues (%d versus %d)\n",
                  getDescription_c(),
                  mTerms[0]->getName(),
                  term->getName(),
                  mTerms[0]->getNumValues(),
                  term->getNumValues());
            diffSizes = true;
         }
      }
      FatalIf(diffSizes, "%s terms are not all the same size.\n", getDescription_c());
   }
   setNumValues(batchWidth);
   return Response::SUCCESS;
}

void ColumnEnergyProbe::addTerm(ProbeInterface *probe) {
   if (probe) {
      mTerms.push_back(probe);
   }
}

void ColumnEnergyProbe::calcValues(double timestamp) {
   ProbeData<double> energy(timestamp, getNumValues(), 0.0);
   for (ProbeInterface *p : mTerms) {
      std::vector<double> const &termValues = p->getValues(timestamp);
      double const coefficient              = p->getCoefficient();
      for (int b = 0; b < getNumValues(); ++b) {
         energy.getValue(b) += coefficient * termValues[b];
      }
   }
   mStoredValues.store(energy);
   setValues(energy);
}

void ColumnEnergyProbe::createComponents(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   // NB: the data members mName and mParams have not been set when createComponents() is called.
   createProbeOutputter(params, defaults, comm);
   createProbeTrigger(params, defaults);
}

void ColumnEnergyProbe::createProbeOutputter(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   mProbeOutputter = std::make_shared<ColumnEnergyOutputter>(params, defaults, comm);
}

void ColumnEnergyProbe::createProbeTrigger(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mProbeTrigger = std::make_shared<ProbeTriggerComponent>(params, defaults);
}

void ColumnEnergyProbe::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   createComponents(params, defaults, comm);
   // createComponents() must be called before the base class's initialize(),
   // because BaseObject::initialize() calls the ioParamsFillGroup() method,
   // which calls each component's ioParamsFillGroup() method.
   ProbeInterface::initialize(params, defaults, comm);
}

void ColumnEnergyProbe::initMessageActionMap() {
   ProbeInterface::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ColProbeOutputStateMessage const>(msgptr);
      return respondColProbeOutputState(castMessage);
   };
   mMessageActionMap.emplace("ColProbeOutputState", action);
}

int ColumnEnergyProbe::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = ProbeInterface::ioParamsFillGroup(ioSwitch);
   mProbeOutputter->ioParamsFillGroup(ioSwitch);
   mProbeTrigger->ioParamsFillGroup(ioSwitch);
   return status;
}

Response::Status ColumnEnergyProbe::outputState(double simTime, double deltaTime) {
   if (mProbeTrigger->needUpdate(simTime, deltaTime)) {
      getValues(simTime);
      // if needed, getValues() will call calcValues(), which will store the result
      // in mStoredValues
   }
   return Response::SUCCESS;
}

Response::Status ColumnEnergyProbe::prepareCheckpointWrite(double simTime) {
   mProbeOutputter->printColumnEnergiesBuffer(mStoredValues);
   mStoredValues.clear();
   return Response::SUCCESS;
}

Response::Status ColumnEnergyProbe::processCheckpointRead(double simTime) {
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
ColumnEnergyProbe::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = ProbeInterface::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer = message->mDataRegistry;
   int localNBatch;
   if (mTerms.empty()) {
      localNBatch = 0;
   }
   else {
      // Check that all terms have the same NumValues.
      int checkNumValues  = PV_SUCCESS;
      localNBatch = mTerms[0]->getNumValues();
      for (auto const *p : mTerms) {
         if (p->getNumValues() != localNBatch) {
            checkNumValues = PV_FAILURE;
            ErrorLog().printf(
                  "%s has energy terms with differing batch widths: "
                  "probe \"%s\" has batch width %d, and probe \"%s\" has batch width %d\n",
                  getDescription_c(),
                  mTerms[0]->getName(),
                  localNBatch,
                  p->getName(),
                  p->getNumValues());
         }
      }
      FatalIf(checkNumValues != PV_SUCCESS, "%s failed.\n", getDescription_c());
   }
   mProbeOutputter->initOutputStreams(checkpointer, localNBatch);
   return Response::SUCCESS;
}

Response::Status ColumnEnergyProbe::respondColProbeOutputState(
      std::shared_ptr<ColProbeOutputStateMessage const>(message)) {
   return outputState(message->mTime, message->mDeltaTime);
}

void ColumnEnergyProbe::setPrintStreams(
      FileStream *printParamsStream, FileStream *printLuaStream) {
   ProbeInterface::setPrintStreams(printParamsStream, printLuaStream);
   setComponentPrintStreams(*mProbeOutputter, printParamsStream, printLuaStream);
   setComponentPrintStreams(*mProbeTrigger, printParamsStream, printLuaStream);
}

} // namespace PV
