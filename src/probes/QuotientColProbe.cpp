/*
 * QuotientColProbe.cpp
 *
 *  Created on: Aug 12, 2015
 *      Author: pschultz
 */

#include "QuotientColProbe.hpp"
#include "columns/HyPerCol.hpp"
#include <limits>

namespace PV {

QuotientColProbe::QuotientColProbe()
      : ColProbe() { // Default constructor to be called by derived classes.
   // They should call QuotientColProbe::initialize from their own initialization
   // routine
   // instead of calling a non-default constructor.
   initialize_base();
} // end QuotientColProbe::QuotientColProbe(const char *)

QuotientColProbe::QuotientColProbe(const char *probename, HyPerCol *hc) : ColProbe() {
   initialize_base();
   initializeQuotientColProbe(probename, hc);
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

int QuotientColProbe::initializeQuotientColProbe(const char *probename, HyPerCol *hc) {
   return ColProbe::initialize(probename, hc);
}

void QuotientColProbe::outputHeader() {
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
   parent->parameters()->ioParamString(
         ioFlag, name, "valueDescription", &valueDescription, "value", true /*warnIfAbsent*/);
}

void QuotientColProbe::ioParam_numerator(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(ioFlag, name, "numerator", &numerator);
}

void QuotientColProbe::ioParam_denominator(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(ioFlag, name, "denominator", &denominator);
}

Response::Status
QuotientColProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = ColProbe::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   numerProbe  = message->lookup<BaseProbe>(std::string(numerator));
   denomProbe  = message->lookup<BaseProbe>(std::string(denominator));
   bool failed = false;
   if (numerProbe == NULL || denomProbe == NULL) {
      failed = true;
      if (parent->columnId() == 0) {
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
   int nNumValues, dNumValues;
   if (!failed) {
      nNumValues = numerProbe->getNumValues();
      dNumValues = denomProbe->getNumValues();
      if (nNumValues != dNumValues) {
         if (parent->columnId() == 0) {
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
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return Response::SUCCESS;
}

void QuotientColProbe::calcValues(double timeValue) {
   int numValues        = this->getNumValues();
   double *valuesBuffer = getValuesBuffer();
   if (parent->simulationTime() == 0.0) {
      for (int b = 0; b < numValues; b++) {
         valuesBuffer[b] = 1.0;
      }
      return;
   }
   double n[numValues];
   numerProbe->getValues(timeValue, n);
   double d[numValues];
   denomProbe->getValues(timeValue, d);
   for (int b = 0; b < numValues; b++) {
      valuesBuffer[b] = n[b] / d[b];
   }
}

double QuotientColProbe::referenceUpdateTime() const { return parent->simulationTime(); }

Response::Status QuotientColProbe::outputState(double timevalue) {
   getValues(timevalue);
   if (mOutputStreams.empty()) {
      return Response::SUCCESS;
   }
   double *valuesBuffer = getValuesBuffer();
   int numValues        = this->getNumValues();
   for (int b = 0; b < numValues; b++) {
      if (isWritingToFile()) {
         output(b) << "\"" << valueDescription << "\",";
      }
      output(b) << timevalue << "," << b << "," << valuesBuffer[b] << std::endl;
   }
   return Response::SUCCESS;
} // end QuotientColProbe::outputState(float, HyPerCol *)

} // end namespace PV
