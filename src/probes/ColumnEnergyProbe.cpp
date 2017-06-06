/*
 * ColumnEnergyProbe.cpp
 *
 *  Created on: Aug 12, 2015
 *      Author: pschultz
 */

#include "ColumnEnergyProbe.hpp"
#include "columns/HyPerCol.hpp"
#include <limits>

namespace PV {

ColumnEnergyProbe::ColumnEnergyProbe()
      : ColProbe() { // Default constructor to be called by derived classes.
   // They should call ColumnEnergyProbe::initialize from their own
   // initialization routine
   // instead of calling a non-default constructor.
   initialize_base();
} // end ColumnEnergyProbe::ColumnEnergyProbe(const char *)

ColumnEnergyProbe::ColumnEnergyProbe(const char *probename, HyPerCol *hc) : ColProbe() {
   initialize_base();
   initializeColumnEnergyProbe(probename, hc);
} // end ColumnEnergyProbe::ColumnEnergyProbe(const char *, HyPerCol *)

ColumnEnergyProbe::~ColumnEnergyProbe() {
   for (int b = 0; b < mOutputBatchElements.size(); b++) {
      delete mOutputBatchElements[b];
   }

   // Don't delete terms[k]; the BaseProbes belong to the layer or connection.
   free(terms);
} // end ColumnEnergyProbe::~ColumnEnergyProbe()

int ColumnEnergyProbe::initialize_base() {
   numTerms = 0;
   terms    = NULL;

   return PV_SUCCESS;
}

int ColumnEnergyProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ColProbe::ioParamsFillGroup(ioFlag);
   ioParam_reductionInterval(ioFlag);
   return status;
}

void ColumnEnergyProbe::ioParam_reductionInterval(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "reductionInterval", &mSkipInterval, mSkipInterval, false /*warnIfAbsent*/);
}

int ColumnEnergyProbe::initializeColumnEnergyProbe(const char *probename, HyPerCol *hc) {
   return ColProbe::initialize(probename, hc);
}

int ColumnEnergyProbe::initOutputStream(const char *filename, Checkpointer *checkpointer) {
   return PV_SUCCESS;
}

int ColumnEnergyProbe::registerData(Checkpointer *checkpointer) {
   MPIBlock const *mpiBlock = checkpointer->getMPIBlock();
   int blockColumnIndex     = mpiBlock->getColumnIndex();
   int blockRowIndex        = mpiBlock->getRowIndex();
   if (blockColumnIndex == 0 and blockRowIndex == 0) {
      int localBatchWidth  = parent->getNBatch();
      int mpiBatchIndex    = mpiBlock->getStartBatch() + mpiBlock->getBatchIndex();
      int localBatchOffset = localBatchWidth * mpiBatchIndex;
      mOutputBatchElements.resize(localBatchWidth);
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
            batchPath = checkpointer->makeOutputPathFilename(batchPath);
            auto fs   = new FileStream(batchPath.c_str(), mode, checkpointer->doesVerifyWrites());
            mOutputBatchElements[b] = fs;
            *fs << "time,index,energy\n";
         }
      }
      else {
         for (int b = 0; b < localBatchWidth; b++) {
            mOutputBatchElements[b] = new PrintStream(PV::getOutputStream());
         }
         *mOutputBatchElements[0] << "Probe_name,time,index,energy\n";
      }
   }
   return PV_SUCCESS;
}

int ColumnEnergyProbe::addTerm(BaseProbe *probe) {
   if (probe == NULL) {
      return PV_FAILURE;
   }
   int status = PV_SUCCESS;
   if (numTerms == 0) {
      int newNumValues = probe->getNumValues();
      if (newNumValues < 0) {
         status = PV_FAILURE;
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: %s cannot be used as a term of the energy "
                  "probe (getNumValue() returned a "
                  "negative number).\n",
                  getDescription_c(),
                  probe->getDescription_c());
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      if (newNumValues != this->getNumValues()) {
         status = setNumValues(newNumValues);
         if (status != PV_SUCCESS) {
            ErrorLog().printf(
                  "%s: unable to allocate memory for %d probe values: %s\n",
                  getDescription_c(),
                  newNumValues,
                  strerror(errno));
            exit(EXIT_FAILURE);
         }
      }
   }
   else {
      if (probe->getNumValues() != this->getNumValues()) {
         if (this->parent->columnId() == 0) {
            ErrorLog().printf(
                  "Failed to add terms to %s:  new probe \"%s\" "
                  "returns %d values, but previous "
                  "probes return %d values\n",
                  getDescription_c(),
                  probe->getName(),
                  probe->getNumValues(),
                  this->getNumValues());
         }
         MPI_Barrier(this->parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
   assert(probe->getNumValues() == getNumValues());
   int newNumTerms = numTerms + (size_t)1;
   if (newNumTerms <= numTerms) {
      if (this->parent->columnId() == 0) {
         ErrorLog().printf(
               "How did you manage to add %zu terms to %s?  "
               "Unable to add any more!\n",
               numTerms,
               getDescription_c());
      }
      MPI_Barrier(this->parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   BaseProbe **newTermsArray =
         (BaseProbe **)realloc(terms, (numTerms + (size_t)1) * sizeof(BaseProbe *));
   if (newTermsArray == NULL) {
      ErrorLog().printf(
            "%s: unable to add term %zu (\"%s\"): %s\n",
            getDescription_c(),
            numTerms + (size_t)1,
            probe->getName(),
            strerror(errno));
      exit(EXIT_FAILURE);
   }
   terms           = newTermsArray;
   terms[numTerms] = probe;
   numTerms        = newNumTerms;
   return PV_SUCCESS;
} // end ColumnEnergyProbe::addTerm(BaseProbe *, double)

bool ColumnEnergyProbe::needRecalc(double timevalue) { return true; }

double ColumnEnergyProbe::referenceUpdateTime() const { return parent->simulationTime(); }

int ColumnEnergyProbe::calcValues(double timevalue) {
   if (mLastTimeValue == timevalue || --mSkipTimer > 0) {
      mLastTimeValue = timevalue;
      return PV_SUCCESS;
   }
   mSkipTimer           = mSkipInterval + 1;
   double *valuesBuffer = getValuesBuffer();
   int numValues        = this->getNumValues();
   memset(valuesBuffer, 0, numValues * sizeof(*valuesBuffer));
   double energy1[numValues];
   for (int n = 0; n < numTerms; n++) {
      BaseProbe *p = terms[n];
      p->getValues(timevalue, energy1);
      double coeff = p->getCoefficient();
      for (int b = 0; b < numValues; b++) {
         valuesBuffer[b] += coeff * energy1[b];
      }
   }
   return PV_SUCCESS;
}

int ColumnEnergyProbe::outputState(double timevalue) {
   getValues(timevalue);
   if (mOutputBatchElements.empty()) {
      return PV_SUCCESS;
   }

   double *valuesBuffer = getValuesBuffer();
   int nbatch           = this->getNumValues();
   pvAssert(nbatch == (int)mOutputBatchElements.size());
   char const *probeOutputFilename = getProbeOutputFilename();
   for (int b = 0; b < nbatch; b++) {
      auto stream = *mOutputBatchElements[b];
      if (probeOutputFilename == nullptr) {
         stream << "\"" << name << "\","; // lack of \n is deliberate
      }
      stream.printf("%10f, %d, %10.9f\n", timevalue, b, valuesBuffer[b]);
      stream.flush();
   }
   return PV_SUCCESS;
} // end ColumnEnergyProbe::outputState(double)

} // end namespace PV
