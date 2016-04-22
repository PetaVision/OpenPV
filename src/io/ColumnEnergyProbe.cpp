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

ColumnEnergyProbe::ColumnEnergyProbe() : ColProbe() { // Default constructor to be called by derived classes.
   // They should call ColumnEnergyProbe::initialize from their own initialization routine
   // instead of calling a non-default constructor.
   initialize_base();
}  // end ColumnEnergyProbe::ColumnEnergyProbe(const char *)

ColumnEnergyProbe::ColumnEnergyProbe(const char * probename, HyPerCol * hc) : ColProbe() {
   initialize_base();
   initializeColumnEnergyProbe(probename, hc);
}  // end ColumnEnergyProbe::ColumnEnergyProbe(const char *, HyPerCol *)

ColumnEnergyProbe::~ColumnEnergyProbe() {
   // Don't delete terms[k]; the BaseProbes belong to the layer or connection.
   free(terms);
}  // end ColumnEnergyProbe::~ColumnEnergyProbe()

int ColumnEnergyProbe::initialize_base() {
   numTerms = 0;
   terms = NULL;

   return PV_SUCCESS;
}

int ColumnEnergyProbe::initializeColumnEnergyProbe(const char * probename, HyPerCol * hc) {
   return ColProbe::initialize(probename, hc);
}

int ColumnEnergyProbe::outputHeader() {
   if (outputstream) {
      fprintf(outputstream->fp, "Probe_name,time,index,energy\n");
   }
   return PV_SUCCESS;
}

int ColumnEnergyProbe::addTerm(BaseProbe * probe) {
   if (probe==NULL) { return PV_FAILURE; }
   int status = PV_SUCCESS;
   if (numTerms==0) {
      int newNumValues = probe->getNumValues();
      if (newNumValues < 0) {
         status = PV_FAILURE;
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: probe \"%s\" cannot be used as a term of the energy probe (getNumValue() returned a negative number).\n",
               getKeyword(), getName(), probe->getName());
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      if (newNumValues != this->getNumValues()) {
         status = setNumValues(newNumValues);
         if (status != PV_SUCCESS) {
            fprintf(stderr, "%s \"%s\" error: unable to allocate memory for %d probe values: %s\n",
                  this->getKeyword(), this->getName(), newNumValues, strerror(errno));
            exit(EXIT_FAILURE);
         }
      }
   }
   else {
      if (probe->getNumValues() != this->getNumValues()) {
         if (this->getParent()->columnId()==0) {
            fprintf(stderr, "Error adding terms to %s \%s\":  new probe \"%s\" returns %d values, but previous probes return %d values\n",
                  getKeyword(), getName(), probe->getName(), probe->getNumValues(), this->getNumValues());
         }
         MPI_Barrier(this->getParent()->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
   assert(probe->getNumValues()==getNumValues());
   int newNumTerms = numTerms+(size_t) 1;
   if (newNumTerms<=numTerms) {
      if (this->getParent()->columnId()==0) {
         fprintf(stderr, "How did you manage to add %zu terms to %s \"%s\"?  Unable to add any more!\n",
               numTerms, getKeyword(), getName());
      }
      MPI_Barrier(this->getParent()->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   BaseProbe ** newTermsArray = (BaseProbe **) realloc(terms, (numTerms+(size_t) 1)*sizeof(BaseProbe *));
   if (newTermsArray==NULL) {
      fprintf(stderr, "%s \"%s\" error: unable to add term %zu (\"%s\"): %s\n",
         getKeyword(), getName(), numTerms+(size_t) 1, probe->getName(),
         strerror(errno));
      exit(EXIT_FAILURE);
   }
   terms = newTermsArray;
   terms[numTerms] = probe;
   numTerms = newNumTerms;
   return PV_SUCCESS;
}  // end ColumnEnergyProbe::addTerm(BaseProbe *, double)

bool ColumnEnergyProbe::needRecalc(double timevalue) {
   return true;
}

double ColumnEnergyProbe::referenceUpdateTime() const {
   return parent->simulationTime();
}

int ColumnEnergyProbe::calcValues(double timevalue) {
   double * valuesBuffer = getValuesBuffer();
   int numValues = this->getNumValues();
   memset(valuesBuffer, 0, numValues*sizeof(*valuesBuffer));
   double energy1[numValues];
   for (int n=0; n<numTerms; n++) {
      BaseProbe * p = terms[n];
      p->getValues(timevalue, energy1);
      double coeff = p->getCoefficient();
      for (int b=0; b<numValues; b++) {
         valuesBuffer[b] += coeff * energy1[b];
      }
   }
   return PV_SUCCESS;
}

int ColumnEnergyProbe::outputState(double timevalue) {
   getValues(timevalue);
   if( this->getParent()->icCommunicator()->commRank() != 0 ) return PV_SUCCESS;
   double * valuesBuffer = getValuesBuffer();
   int nbatch = this->getNumValues();
   for(int b = 0; b < nbatch; b++){
      if (outputstream->fp == stdout || outputstream->fp == stderr) {
         fprintf(outputstream->fp,"\"%s\",", name); // lack of \n is deliberate: fprintf immediately below completes the line
      }
      fprintf(outputstream->fp, "%f,%d,%f\n",
            timevalue, b,valuesBuffer[b]);
   }
   fflush(outputstream->fp);
   return PV_SUCCESS;
}  // end ColumnEnergyProbe::outputState(float, HyPerCol *)

BaseObject * createColumnEnergyProbe(char const * name, HyPerCol * hc) {
   return hc ? new ColumnEnergyProbe(name, hc) : NULL;
}

}  // end namespace PV
