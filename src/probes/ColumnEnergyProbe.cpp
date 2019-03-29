/*
 * ColumnEnergyProbe.cpp
 *
 *  Created on: Aug 12, 2015
 *      Author: pschultz
 */

#include "ColumnEnergyProbe.hpp"
#include <limits>

namespace PV {

ColumnEnergyProbe::ColumnEnergyProbe()
      : ColProbe() { // Default constructor to be called by derived classes.
   // They should call ColumnEnergyProbe::initialize from their own
   // initialization routine
   // instead of calling a non-default constructor.
   initialize_base();
} // end ColumnEnergyProbe::ColumnEnergyProbe(const char *)

ColumnEnergyProbe::ColumnEnergyProbe(
      const char *probename,
      PVParams *params,
      Communicator const *comm)
      : ColProbe() {
   initialize_base();
   initialize(probename, params, comm);
} // end ColumnEnergyProbe::ColumnEnergyProbe(const char *, HyPerCol *)

ColumnEnergyProbe::~ColumnEnergyProbe() {
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
   parameters()->ioParamValue(
         ioFlag, name, "reductionInterval", &mSkipInterval, mSkipInterval, false /*warnIfAbsent*/);
}

void ColumnEnergyProbe::initialize(
      const char *probename,
      PVParams *params,
      Communicator const *comm) {
   ColProbe::initialize(probename, params, comm);
}

void ColumnEnergyProbe::outputHeader() {
   if (isWritingToFile()) {
      for (auto &s : mOutputStreams) {
         *s << "time,index,energy\n";
      }
   }
   else {
      if (!mOutputStreams.empty()) {
         *mOutputStreams[0] << "Probe_name,time,index,energy\n";
      }
   }
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
         if (mCommunicator->commRank() == 0) {
            ErrorLog().printf(
                  "%s: %s cannot be used as a term of the energy "
                  "probe (getNumValue() returned a "
                  "negative number).\n",
                  getDescription_c(),
                  probe->getDescription_c());
         }
         MPI_Barrier(mCommunicator->communicator());
         exit(EXIT_FAILURE);
      }
      if (newNumValues != this->getNumValues()) {
         setNumValues(newNumValues);
      }
   }
   else {
      if (probe->getNumValues() != this->getNumValues()) {
         if (this->mCommunicator->commRank() == 0) {
            ErrorLog().printf(
                  "Failed to add terms to %s:  new probe \"%s\" "
                  "returns %d values, but previous "
                  "probes return %d values\n",
                  getDescription_c(),
                  probe->getName(),
                  probe->getNumValues(),
                  this->getNumValues());
         }
         MPI_Barrier(this->mCommunicator->communicator());
         exit(EXIT_FAILURE);
      }
   }
   assert(probe->getNumValues() == getNumValues());
   auto newNumTerms = numTerms + (std::size_t)1;
   if (newNumTerms <= numTerms) {
      if (this->mCommunicator->commRank() == 0) {
         ErrorLog().printf(
               "How did you manage to add %zu terms to %s?  "
               "Unable to add any more!\n",
               numTerms,
               getDescription_c());
      }
      MPI_Barrier(this->mCommunicator->communicator());
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
   return status;
} // end ColumnEnergyProbe::addTerm(BaseProbe *, double)

bool ColumnEnergyProbe::needRecalc(double timevalue) { return true; }

double ColumnEnergyProbe::referenceUpdateTime(double simTime) const { return simTime; }

void ColumnEnergyProbe::calcValues(double timevalue) {
   if (mLastTimeValue == timevalue || --mSkipTimer > 0) {
      mLastTimeValue = timevalue;
      return;
   }
   mSkipTimer           = mSkipInterval + 1;
   double *valuesBuffer = getValuesBuffer();
   int numValues        = this->getNumValues();
   memset(valuesBuffer, 0, numValues * sizeof(*valuesBuffer));
   double energy1[numValues];
   for (std::size_t n = 0; n < numTerms; n++) {
      BaseProbe *p = terms[n];
      p->getValues(timevalue, energy1);
      double coeff = p->getCoefficient();
      for (int b = 0; b < numValues; b++) {
         valuesBuffer[b] += coeff * energy1[b];
      }
   }
}

Response::Status ColumnEnergyProbe::outputState(double simTime, double deltaTime) {
   getValues(simTime);
   if (mOutputStreams.empty()) {
      return Response::SUCCESS;
   }

   double *valuesBuffer = getValuesBuffer();
   int nbatch           = this->getNumValues();
   pvAssert(nbatch == (int)mOutputStreams.size());
   for (int b = 0; b < nbatch; b++) {
      auto stream = *mOutputStreams[b];
      if (!isWritingToFile()) {
         stream << "\"" << name << "\","; // lack of \n is deliberate
      }
      stream.printf("%10f, %d, %10.9f\n", simTime, b, valuesBuffer[b]);
      stream.flush();
   }
   return Response::SUCCESS;
} // end ColumnEnergyProbe::outputState(double)

} // end namespace PV
