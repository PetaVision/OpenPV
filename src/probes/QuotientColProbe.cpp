/*
 * QuotientColProbe.cpp
 *
 *  Created on: Aug 12, 2015
 *      Author: pschultz
 */

#include "QuotientColProbe.hpp"
#include <limits>

namespace PV {

QuotientColProbe::QuotientColProbe()
      : ColProbe() { // Default constructor to be called by derived classes.
   // They should call QuotientColProbe::initialize from their own initialization
   // routine
   // instead of calling a non-default constructor.
   initialize_base();
} // end QuotientColProbe::QuotientColProbe(const char *)

QuotientColProbe::QuotientColProbe(
      const char *probename,
      PVParams *params,
      Communicator const *comm)
      : ColProbe() {
   initialize_base();
   initialize(probename, params, comm);
}

QuotientColProbe::~QuotientColProbe() {
   free(valueDescription);
   free(numerator);
   free(denominator);
   // Don't free numerProbe or denomProbe; they don't belong to the
   // QuotientColProbe.
}

int QuotientColProbe::initialize_base() {
   valueDescription = NULL;
   numerator        = NULL;
   denominator      = NULL;
   numerProbe       = NULL;
   denomProbe       = NULL;
   return PV_SUCCESS;
}

void QuotientColProbe::initialize(
      const char *probename,
      PVParams *params,
      Communicator const *comm) {
   ColProbe::initialize(probename, params, comm);
}

void QuotientColProbe::outputHeader(Checkpointer *checkpointer) {
   for (auto &s : mOutputStreams) {
      *s << "Probe_name,time,index," << valueDescription;
   }
}

int QuotientColProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ColProbe::ioParamsFillGroup(ioFlag);
   ioParam_valueDescription(ioFlag);
   ioParam_numerator(ioFlag);
   ioParam_denominator(ioFlag);
   return status;
}

void QuotientColProbe::ioParam_valueDescription(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag, name, "valueDescription", &valueDescription, "value", true /*warnIfAbsent*/);
}

void QuotientColProbe::ioParam_numerator(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, name, "numerator", &numerator);
}

void QuotientColProbe::ioParam_denominator(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, name, "denominator", &denominator);
}

Response::Status
QuotientColProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = ColProbe::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *objectTable = message->mObjectTable;
   numerProbe        = objectTable->findObject<BaseProbe>(numerator);
   denomProbe        = objectTable->findObject<BaseProbe>(denominator);
   bool failed       = false;
   if (numerProbe == NULL || denomProbe == NULL) {
      failed = true;
      if (mCommunicator->commRank() == 0) {
         if (numerProbe == NULL) {
            ErrorLog().printf(
                  "%s: numerator probe \"%s\" could not be found.\n",
                  getDescription_c(),
                  numerator);
         }
         if (denomProbe == NULL) {
            ErrorLog().printf(
                  "%s: denominator probe \"%s\" could not be found.\n",
                  getDescription_c(),
                  denominator);
         }
      }
   }
   if (!numerProbe->getInitInfoCommunicatedFlag() or !denomProbe->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   int nNumValues, dNumValues;
   if (!failed) {
      nNumValues = numerProbe->getNumValues();
      dNumValues = denomProbe->getNumValues();
      if (nNumValues != dNumValues) {
         if (mCommunicator->commRank() == 0) {
            ErrorLog().printf(
                  "%s: numerator probe \"%s\" and denominator "
                  "probe \"%s\" have differing numbers "
                  "of values (%d vs. %d)\n",
                  getDescription_c(),
                  numerator,
                  denominator,
                  nNumValues,
                  dNumValues);
         }
         failed = true;
      }
   }
   if (!failed) {
      setNumValues(nNumValues);
   }
   if (failed) {
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
   return Response::SUCCESS;
}

void QuotientColProbe::calcValues(double timeValue) {
   int numValues        = this->getNumValues();
   auto &valuesVector = getProbeValues();
   pvAssert(numValues == static_cast<int>(valuesVector.size()));
   if (timeValue == 0.0) {
      for (int b = 0; b < numValues; b++) {
         valuesVector[b] = 1.0;
      }
      return;
   }
   double n[numValues];
   numerProbe->getValues(timeValue, n);
   double d[numValues];
   denomProbe->getValues(timeValue, d);
   for (int b = 0; b < numValues; b++) {
      valuesVector[b] = n[b] / d[b];
   }
}

double QuotientColProbe::referenceUpdateTime(double simTime) const { return simTime; }

Response::Status QuotientColProbe::outputState(double simTime, double deltaTime) {
   getValues(simTime);
   if (mOutputStreams.empty()) {
      return Response::SUCCESS;
   }
   auto &valuesVector = getProbeValues();
   int numValues        = this->getNumValues();
   pvAssert(numValues == static_cast<int>(valuesVector.size()));
   for (int b = 0; b < numValues; b++) {
      if (isWritingToFile()) {
         output(b) << "\"" << valueDescription << "\","; // lack of \n is deliberate
      }
      output(b) << simTime << "," << b << "," << valuesVector[b] << std::endl;
   }
   return Response::SUCCESS;
} // end QuotientColProbe::outputState(float, HyPerCol *)

Response::Status QuotientColProbe::outputStateStats(double simTime, double deltaTime) {
   getValues(simTime);
   auto &valuesVector = getProbeValues();
   int nbatch           = getNumValues();
   pvAssert(nbatch == static_cast<int>(valuesVector.size()));
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
      if (isWritingToFile()) {
         output(0) << "\"" << valueDescription << "\","; // lack of \n is deliberate
      }
      output(0).printf("t=%10f, min=%10.9f, max=%10.9f, mean=%10.9f\n", simTime, min, max, mean);
   }
   else {
      pvAssert(mCommunicator->globalCommRank() != 0);
   }
   return Response::SUCCESS;
}

} // end namespace PV
