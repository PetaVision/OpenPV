/*
 * ColumnEnergyProbe.cpp
 *
 *  Created on: Aug 12, 2015
 *      Author: pschultz
 */

#include "ColumnEnergyProbe.hpp"
#include "BaseProbe.hpp"
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

int ColumnEnergyProbe::addTerm(BaseProbe * probe, double coefficient, size_t vectorSize) {
   if (probe==NULL) { return PV_FAILURE; }
   if (numTerms>0 && vectorSize != this->vectorSize) {
      if (this->getParent()->columnId()==0) {
         fprintf(stderr, "Error adding terms to %s \%s\": vector size %zu of new probe \"%s\" does not agree with existing vector size %zu\n",
               getKeyword(), getName(), vectorSize, probe->getName(), this->vectorSize);
      }
      exit(EXIT_FAILURE);
   }
   this->vectorSize = vectorSize;
   size_t newNumTerms = numTerms+(size_t) 1;
   if (newNumTerms<=numTerms) {
      if (this->getParent()->columnId()==0) {
         fprintf(stderr, "How did you manage to add %zu terms to %s \"%s\"?  Unable to add any more!\n",
               numTerms, getKeyword(), getName());
      }
      MPI_Barrier(this->getParent()->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   energyTerm * newTermsArray = (energyTerm *) realloc(terms, (numTerms+(size_t) 1)*sizeof(energyTerm));
   if (newTermsArray==NULL) {
      fprintf(stderr, "%s \"%s\" error: unable to add term %zu (\"%s\"): %s\n",
         getKeyword(), getName(), numTerms+(size_t) 1, probe->getName(),
         strerror(errno));
      exit(EXIT_FAILURE);
   }
   terms = newTermsArray;
   terms[numTerms].probe = probe;
   terms[numTerms].coeff = coefficient;
   numTerms = newNumTerms;
   return PV_SUCCESS;
}  // end ColumnEnergyProbe::addTerm(BaseProbe *, double)

int ColumnEnergyProbe::getValues(double timevalue, std::vector<double> * values) {
   values->assign(vectorSize, 0.0);
   for (int n=0; n<numTerms; n++) {
      std::vector<double> energy1;
      energyTerm * p = &terms[n];
      p->probe->getValues(timevalue, &energy1);
      for (int b=0; b<vectorSize; b++) {
         values->at(b) += p->coeff * energy1.at(b);
      }
   }
   return PV_SUCCESS;
}

double ColumnEnergyProbe::getValue(double timevalue, int index) {
   if (index<0 || index>=vectorSize) {
      std::numeric_limits<double>::signaling_NaN();
   }
   double sum = 0;
   for (int n=0; n<numTerms; n++) {
      energyTerm * p = &terms[n];
      sum += p->coeff * p->probe->getValue(timevalue, index);
   }
   return sum;
}  // end ColumnEnergyProbe::evaluate(float)

int ColumnEnergyProbe::outputState(double timevalue) {
   std::vector<double> energy;
   getValues(timevalue, &energy);
   if( this->getParent()->icCommunicator()->commRank() != 0 ) return PV_SUCCESS;
   int nbatch = this->getParent()->getNBatch();
   for(int b = 0; b < nbatch; b++){
      fprintf(outputstream->fp, "\"%s\",%f,%d,%f\n",
            this->getName(), timevalue, b, energy.at(b));
   }
   fflush(outputstream->fp);
   return PV_SUCCESS;
}  // end ColumnEnergyProbe::outputState(float, HyPerCol *)

}  // end namespace PV
