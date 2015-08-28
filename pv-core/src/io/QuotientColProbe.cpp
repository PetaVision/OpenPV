/*
 * QuotientColProbe.cpp
 *
 *  Created on: Aug 12, 2015
 *      Author: pschultz
 */

#include "QuotientColProbe.hpp"
#include "BaseProbe.hpp"
#include <limits>

namespace PV {

QuotientColProbe::QuotientColProbe() : ColProbe() { // Default constructor to be called by derived classes.
   // They should call QuotientColProbe::initialize from their own initialization routine
   // instead of calling a non-default constructor.
   initialize_base();
}  // end QuotientColProbe::QuotientColProbe(const char *)

QuotientColProbe::QuotientColProbe(const char * probename, HyPerCol * hc) : ColProbe() {
   initialize_base();
   initializeQuotientColProbe(probename, hc);
}

QuotientColProbe::~QuotientColProbe() {
   free(valueDescription);
   free(numerator);
   free(denominator);
   // Don't free numerProbe or denomProbe; they don't belong to the QuotientColProbe.
}

int QuotientColProbe::initialize_base() {
   valueDescription = NULL;
   numerator = NULL;
   denominator = NULL;
   numerProbe = NULL;
   denomProbe = NULL;
   return PV_SUCCESS;
}

int QuotientColProbe::initializeQuotientColProbe(const char * probename, HyPerCol * hc) {
   return ColProbe::initialize(probename, hc);
}

int QuotientColProbe::outputHeader() {
   if (outputstream) {
      fprintf(outputstream->fp, "Probe_name,time,index,%s\n", valueDescription);
   }
   return PV_SUCCESS;
}

int QuotientColProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ColProbe::ioParamsFillGroup(ioFlag);
   ioParam_valueDescription(ioFlag);
   ioParam_numerator(ioFlag);
   ioParam_denominator(ioFlag);
   return status;
}

void QuotientColProbe::ioParam_valueDescription(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "valueDescription", &valueDescription, "value", true/*warnIfAbsent*/);
}

void QuotientColProbe::ioParam_numerator(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "numerator", &numerator);
}

void QuotientColProbe::ioParam_denominator(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "denominator", &denominator);
}

int QuotientColProbe::communicateInitInfo() {
   int status = ColProbe::communicateInitInfo();
   numerProbe = findProbe(numerator);
   denomProbe = findProbe(denominator);
   if (numerProbe==NULL || denomProbe==NULL) {
      status = PV_FAILURE;
      if (parent->columnId()==0) {
         if (numerProbe==NULL) {
            fprintf(stderr, "%s \"%s\" error: numerator probe \"%s\" could not be found.\n", getKeyword(), getName(), numerator);
         }
         if (denomProbe==NULL) {
            fprintf(stderr, "%s \"%s\" error: denominator probe \"%s\" could not be found.\n", getKeyword(), getName(), denominator);
         }
      }
   }
   if (status == PV_SUCCESS) {
      int nNumValues = numerProbe->getNumValues();
      int dNumValues = denomProbe->getNumValues();
      if (nNumValues != dNumValues) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: numerator probe \"%s\" and denominator probe \"%s\" have differing numbers of values (%d vs. %d)\n",
                  getKeyword(), getName(), numerator, denominator, nNumValues, dNumValues);
         }
         MPI_Barrier(this->getParent()->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      status = setNumValues(nNumValues);
      if (status != PV_SUCCESS) {
         fprintf(stderr, "%s \"%s\" error: unable to allocate memory for %d values: %s\n",
               this->getKeyword(), this->getName(), nNumValues, strerror(errno));
         exit(EXIT_FAILURE);
      }
   }
   if (status != PV_SUCCESS) {
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return status;
}

BaseProbe * QuotientColProbe::findProbe(char const * probeName) {
   // Search the ColProbes
   for (int p=0; p<parent->numberOfProbes(); p++) {
      BaseProbe * probe = parent->getColProbe(p);
      if (!strcmp(probe->getName(), probeName)) { return probe; }
   }
   // Search the layer probes
   for (int l=0; l<parent->numberOfLayers(); l++) {
      HyPerLayer * layer = parent->getLayer(l);
      for (int p=0; p < layer->getNumProbes(); p++) {
         BaseProbe * probe = layer->getProbe(p);
         if (!strcmp(probe->getName(), probeName)) { return probe; }
      }
   }
   // Search the connection probes
   for (int c=0; c<parent->numberOfLayers(); c++) {
      BaseConnection * conn = parent->getConnection(c);
      for (int p=0; p < conn->getNumProbes(); p++) {
         BaseProbe * probe = (BaseProbe *) conn->getProbe(p);
         if (!strcmp(probe->getName(), probeName)) { return probe; }
      }
   }
   // If you reach here, no such probe was found.
   return NULL;
}

int QuotientColProbe::calcValues(double timeValue) {
   int numValues = this->getNumValues();
   double n[numValues];
   numerProbe->getValues(timeValue, n);
   double d[numValues];
   denomProbe->getValues(timeValue, d);
   double * valuesBuffer = getValuesBuffer();
   for (int b=0; b<numValues; b++) {
      valuesBuffer[b] = n[b]/d[b];
   }
   return PV_SUCCESS;
}

int QuotientColProbe::outputState(double timevalue) {
   getValues(timevalue);
   if( this->getParent()->icCommunicator()->commRank() != 0 ) return PV_SUCCESS;
   double * valuesBuffer = getValuesBuffer();
   int numValues = this->getNumValues();
   for(int b = 0; b < numValues; b++){
      fprintf(outputstream->fp, "%s,%f,%d,%f\n",
            this->valueDescription, timevalue, b, valuesBuffer[b]);
   }
   fflush(outputstream->fp);
   return PV_SUCCESS;
}  // end QuotientColProbe::outputState(float, HyPerCol *)

}  // end namespace PV
