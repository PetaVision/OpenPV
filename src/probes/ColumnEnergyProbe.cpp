#include "ColumnEnergyProbe.hpp"
#include "include/pv_common.h"
#include "observerpattern/BaseMessage.hpp"
#include "probes/ProbeData.hpp"
#include "utils/PVLog.hpp"
#include <functional>

namespace PV {

ColumnEnergyProbe::ColumnEnergyProbe(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

Response::Status ColumnEnergyProbe::allocateDataStructures() {
   Response::Status status = ProbeInterface::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
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
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   // NB: the data members mName and mParams have not been set when createComponents() is called.
   createProbeOutputter(name, params, comm);
   createProbeTrigger(name, params);
}

void ColumnEnergyProbe::createProbeOutputter(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mProbeOutputter = std::make_shared<ColumnEnergyOutputter>(name, params, comm);
}

void ColumnEnergyProbe::createProbeTrigger(char const *name, PVParams *params) {
   mProbeTrigger = std::make_shared<ProbeTriggerComponent>(name, params);
}

void ColumnEnergyProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   createComponents(name, params, comm);
   // createComponents() must be called before the base class's initialize(),
   // because BaseObject::initialize() calls the ioParamsFillGroup() method,
   // which calls each component's ioParamsFillGroup() method.
   ProbeInterface::initialize(name, params, comm);
}

void ColumnEnergyProbe::initMessageActionMap() {
   ProbeInterface::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ColProbeOutputStateMessage const>(msgptr);
      return respondColProbeOutputState(castMessage);
   };
   mMessageActionMap.emplace("ColProbeOutputState", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ProbeWriteParamsMessage const>(msgptr);
      return respondProbeWriteParams(castMessage);
   };
   mMessageActionMap.emplace("ProbeWriteParams", action);
}

int ColumnEnergyProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ProbeInterface::ioParamsFillGroup(ioFlag);
   mProbeOutputter->ioParamsFillGroup(ioFlag);
   mProbeTrigger->ioParamsFillGroup(ioFlag);
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
      int status  = PV_SUCCESS;
      localNBatch = mTerms[0]->getNumValues();
      for (auto const *p : mTerms) {
         if (p->getNumValues() != localNBatch) {
            status = PV_FAILURE;
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
      FatalIf(status != PV_SUCCESS, "%s failed.\n", getDescription_c());
   }
   mProbeOutputter->initOutputStreams(checkpointer, localNBatch);
   return Response::SUCCESS;
}

Response::Status ColumnEnergyProbe::respondColProbeOutputState(
      std::shared_ptr<ColProbeOutputStateMessage const>(message)) {
   return outputState(message->mTime, message->mDeltaTime);
}

Response::Status
ColumnEnergyProbe::respondProbeWriteParams(std::shared_ptr<ProbeWriteParamsMessage const> message) {
   writeParams();
   return Response::SUCCESS;
}

} // namespace PV
