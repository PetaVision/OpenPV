/*
 * BaseProbe.cpp
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#include "BaseProbe.hpp"
#include "ColumnEnergyProbe.hpp"
#include "layers/HyPerLayer.hpp"
#include <float.h>
#include <limits>

namespace PV {

BaseProbe::BaseProbe() {
   initialize_base();
   // Derived classes of BaseProbe should call BaseProbe::initialize themselves.
}

BaseProbe::~BaseProbe() {
   for (auto &s : mOutputStreams) {
      delete s;
   }
   mOutputStreams.clear();
   free(targetName);
   targetName = NULL;
   free(msgparams);
   msgparams = NULL;
   free(msgstring);
   msgstring = NULL;
   free(probeOutputFilename);
   probeOutputFilename = NULL;
   if (triggerLayerName) {
      free(triggerLayerName);
      triggerLayerName = NULL;
   }
   free(energyProbe);
   free(probeValues);
}

int BaseProbe::initialize_base() {
   targetName          = NULL;
   msgparams           = NULL;
   msgstring           = NULL;
   textOutputFlag      = true;
   probeOutputFilename = NULL;
   triggerFlag         = false;
   triggerLayerName    = NULL;
   triggerLayer        = NULL;
   triggerOffset       = 0;
   energyProbe         = NULL;
   coefficient         = 1.0;
   numValues           = 0;
   probeValues         = NULL;
   lastUpdateTime      = 0.0;
   return PV_SUCCESS;
}

/**
 * @filename
 * @layer
 */
int BaseProbe::initialize(const char *name, HyPerCol *hc) {
   int status = BaseObject::initialize(name, hc);
   if (status != PV_SUCCESS) {
      return status;
   }
   readParams();
   initNumValues();
   return status;
}

void BaseProbe::setObjectType() { mObjectType = lookupKeyword(); }

int BaseProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_targetName(ioFlag);
   ioParam_message(ioFlag);
   ioParam_textOutputFlag(ioFlag);
   ioParam_probeOutputFile(ioFlag);
   ioParam_triggerLayerName(ioFlag);
   ioParam_triggerFlag(ioFlag);
   ioParam_triggerOffset(ioFlag);
   ioParam_energyProbe(ioFlag);
   ioParam_coefficient(ioFlag);
   return PV_SUCCESS;
}

void BaseProbe::ioParam_targetName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(ioFlag, name, "targetName", &targetName);
}

void BaseProbe::ioParam_message(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(
         ioFlag, name, "message", &msgparams, NULL, false /*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      initMessage(msgparams);
   }
}

void BaseProbe::ioParam_energyProbe(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(
         ioFlag, name, "energyProbe", &energyProbe, NULL, false /*warnIfAbsent*/);
}

void BaseProbe::ioParam_coefficient(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "energyProbe"));
   if (energyProbe && energyProbe[0]) {
      parent->parameters()->ioParamValue(
            ioFlag, name, "coefficient", &coefficient, coefficient, true /*warnIfAbsent*/);
   }
}

void BaseProbe::ioParam_textOutputFlag(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "textOutputFlag", &textOutputFlag, textOutputFlag);
}

void BaseProbe::ioParam_probeOutputFile(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "textOutputFlag"));
   if (textOutputFlag) {
      parent->parameters()->ioParamString(
            ioFlag, name, "probeOutputFile", &probeOutputFilename, NULL, false /*warnIfAbsent*/);
   }
}

void BaseProbe::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(
         ioFlag, name, "triggerLayerName", &triggerLayerName, NULL, false /*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      triggerFlag = (triggerLayerName != NULL && triggerLayerName[0] != '\0');
   }
}

// triggerFlag was deprecated Oct 7, 2015, and marked obsolete Jun 9, 2017.
// Setting triggerLayerName to a nonempty string has the effect of triggerFlag=true, and
// setting triggerLayerName to NULL or "" has the effect of triggerFlag=false.
// For a reasonable fade-out time, it is an error for triggerFlag to be defined in params.
void BaseProbe::ioParam_triggerFlag(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (ioFlag == PARAMS_IO_READ && parent->parameters()->present(name, "triggerFlag")) {
      bool flagFromParams = false;
      parent->parameters()->ioParamValue(
            ioFlag, name, "triggerFlag", &flagFromParams, flagFromParams);
      if (parent->columnId() == 0) {
         Fatal(triggerFlagDeprecated);
         triggerFlagDeprecated.printf(
               "%s: triggerFlag is obsolete for probes.\n", getDescription_c());
         triggerFlagDeprecated.printf(
               "   If triggerLayerName is a nonempty string, triggering will be on;\n");
         triggerFlagDeprecated.printf(
               "   if triggerLayerName is empty or null, triggering will be off.\n");
      }
   }
}

void BaseProbe::ioParam_triggerOffset(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "triggerFlag"));
   if (triggerFlag) {
      parent->parameters()->ioParamValue(
            ioFlag, name, "triggerOffset", &triggerOffset, triggerOffset);
      if (triggerOffset < 0) {
         Fatal().printf(
               "%s \"%s\" error in rank %d process: TriggerOffset (%f) "
               "must be positive\n",
               parent->parameters()->groupKeywordFromName(name),
               name,
               parent->columnId(),
               triggerOffset);
      }
   }
}

void BaseProbe::initOutputStreams(const char *filename, Checkpointer *checkpointer) {
   MPIBlock const *mpiBlock = checkpointer->getMPIBlock();
   int blockColumnIndex     = mpiBlock->getColumnIndex();
   int blockRowIndex        = mpiBlock->getRowIndex();
   if (blockColumnIndex == 0 and blockRowIndex == 0) {
      int localBatchWidth  = parent->getNBatch();
      int mpiBatchIndex    = mpiBlock->getStartBatch() + mpiBlock->getBatchIndex();
      int localBatchOffset = localBatchWidth * mpiBatchIndex;
      mOutputStreams.resize(localBatchWidth);
      char const *probeOutputFilename = getProbeOutputFilename();
      if (probeOutputFilename) {
         std::string path(probeOutputFilename);
         auto extensionStart = path.rfind('.');
         std::string extension;
         if (extensionStart != std::string::npos) {
            extension = path.substr(extensionStart);
            path      = path.substr(0, extensionStart);
         }
         std::ios_base::openmode mode = std::ios_base::out;
         if (!checkpointer->getCheckpointReadDirectory().empty()) {
            mode |= std::ios_base::app;
         }
         for (int b = 0; b < localBatchWidth; b++) {
            int globalBatchIndex         = b + localBatchOffset;
            std::string batchPath        = path;
            std::string batchIndexString = std::to_string(globalBatchIndex);
            batchPath.append("_batchElement_").append(batchIndexString).append(extension);
            if (batchPath[0] != '/') {
               batchPath = checkpointer->makeOutputPathFilename(batchPath);
            }
            auto fs = new FileStream(batchPath.c_str(), mode, checkpointer->doesVerifyWrites());
            mOutputStreams[b] = fs;
         }
      }
      else {
         for (int b = 0; b < localBatchWidth; b++) {
            mOutputStreams[b] = new PrintStream(PV::getOutputStream());
         }
      }
   }
   else {
      mOutputStreams.clear();
   }
}

void BaseProbe::initNumValues() { setNumValues(parent->getNBatch()); }

void BaseProbe::setNumValues(int n) {
   if (n > 0) {
      double *newValuesBuffer = (double *)realloc(probeValues, (size_t)n * sizeof(*probeValues));
      FatalIf(
            newValuesBuffer == nullptr,
            "%s unable to set number of values to %d\n",
            getDescription_c(),
            n);
      // realloc() succeeded
      probeValues = newValuesBuffer;
      numValues   = n;
   }
   else {
      free(probeValues);
      probeValues = nullptr;
   }
}

Response::Status
BaseProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   // Set up triggering.
   if (triggerFlag) {
      triggerLayer = message->lookup<HyPerLayer>(std::string(triggerLayerName));
      if (triggerLayer == NULL) {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s \"%s\": triggerLayer \"%s\" is not a layer in the HyPerCol.\n",
                  parent->parameters()->groupKeywordFromName(name),
                  name,
                  triggerLayerName);
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }

   // Add the probe to the ColumnEnergyProbe, if there is one.
   if (energyProbe && energyProbe[0]) {
      ColumnEnergyProbe *probe = message->lookup<ColumnEnergyProbe>(std::string(energyProbe));
      if (probe == NULL) {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s \"%s\": energyProbe \"%s\" is not a ColumnEnergyProbe in the "
                  "column.\n",
                  parent->parameters()->groupKeywordFromName(getName()),
                  getName(),
                  energyProbe);
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      int termAdded = probe->addTerm(this);
      FatalIf(
            termAdded != PV_SUCCESS,
            "Failed to add %s to %s.\n",
            getDescription_c(),
            probe->getDescription_c());
   }
   return Response::SUCCESS;
}

int BaseProbe::initMessage(const char *msg) {
   assert(msgstring == NULL);
   int status = PV_SUCCESS;
   if (msg != NULL && msg[0] != '\0') {
      size_t msglen   = strlen(msg);
      this->msgstring = (char *)calloc(
            msglen + 2,
            sizeof(char)); // Allocate room for colon plus null terminator
      if (this->msgstring) {
         memcpy(this->msgstring, msg, msglen);
         this->msgstring[msglen]     = ':';
         this->msgstring[msglen + 1] = '\0';
      }
   }
   else {
      this->msgstring = (char *)calloc(1, sizeof(char));
      if (this->msgstring) {
         this->msgstring[0] = '\0';
      }
   }
   if (!this->msgstring) {
      ErrorLog().printf(
            "%s \"%s\": Unable to allocate memory for probe's message.\n",
            parent->parameters()->groupKeywordFromName(name),
            name);
      status = PV_FAILURE;
   }
   assert(status == PV_SUCCESS);
   return status;
}

bool BaseProbe::needUpdate(double simTime, double dt) {
   if (triggerFlag) {
      return triggerLayer->needUpdate(simTime + triggerOffset, dt);
   }
   return true;
}

Response::Status BaseProbe::registerData(Checkpointer *checkpointer) {
   auto status = BaseObject::registerData(checkpointer);
   if (!Response::completed(status)) {
      return status;
   }
   initOutputStreams(probeOutputFilename, checkpointer);
   return Response::SUCCESS;
}

void BaseProbe::getValues(double timevalue) {
   if (needRecalc(timevalue)) {
      calcValues(timevalue);
      lastUpdateTime = referenceUpdateTime();
   }
}

void BaseProbe::getValues(double timevalue, double *values) {
   getValues(timevalue);
   memcpy(values, probeValues, sizeof(*probeValues) * (size_t)getNumValues());
}

void BaseProbe::getValues(double timevalue, std::vector<double> *valuesVector) {
   valuesVector->resize(this->getNumValues());
   getValues(timevalue, &valuesVector->front());
}

double BaseProbe::getValue(double timevalue, int index) {
   if (index < 0 || index >= getNumValues()) {
      return std::numeric_limits<double>::signaling_NaN();
   }
   else {
      if (needRecalc(timevalue)) {
         getValues(timevalue);
      }
   }
   return probeValues[index];
}

Response::Status BaseProbe::outputStateWrapper(double timef, double dt) {
   auto status = Response::NO_ACTION;
   if (textOutputFlag && needUpdate(timef, dt)) {
      status = outputState(timef);
   }
   return status;
}

} // namespace PV
