/*
 * AbstractNormProbe.cpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#include "AbstractNormProbe.hpp"
#include "../columns/HyPerCol.hpp"
#include <limits>

namespace PV {

AbstractNormProbe::AbstractNormProbe() : LayerProbe() {
   initAbstractNormProbe_base();
}

AbstractNormProbe::AbstractNormProbe(const char * probeName, HyPerCol * hc) : LayerProbe()
{
   initAbstractNormProbe_base();
   initAbstractNormProbe(probeName, hc);
}

AbstractNormProbe::~AbstractNormProbe() {
   free(normDescription); normDescription = NULL;
}

int AbstractNormProbe::initAbstractNormProbe(const char * probeName, HyPerCol * hc) {
   int status = LayerProbe::initialize(probeName, hc);
   if (status == PV_SUCCESS) {
      status = setNormDescription();
   }
   return status;
}

int AbstractNormProbe::getValues(double timevalue, std::vector<double> * values) {
   if (values==NULL) { return PV_FAILURE; }
   int nBatch = getParent()->getNBatch();
   values->resize(nBatch); // Should we test if values->size()==nBatch before resizing?
   for (int b=0; b<nBatch; b++) {
      values->at(b) = getValueInternal(timevalue, b);
   }
   MPI_Allreduce(MPI_IN_PLACE, &values->front(), nBatch, MPI_DOUBLE, MPI_SUM, getParent()->icCommunicator()->communicator());
   return PV_SUCCESS;
}
   
double AbstractNormProbe::getValue(double timevalue, int index) {
   if (index>=0 && index < getParent()->getNBatch()) {
      double norm = getValueInternal(timevalue, index);
      MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, getParent()->icCommunicator()->communicator());
      return norm;
   }
   else {
      return std::numeric_limits<double>::signaling_NaN();
   }
}

int AbstractNormProbe::setNormDescription() {
    return setNormDescriptionToString("norm");
}

int AbstractNormProbe::setNormDescriptionToString(char const * s) {
    normDescription = strdup(s);
    return normDescription ? PV_SUCCESS : PV_FAILURE;
}

int AbstractNormProbe::outputState(double timevalue) {
   std::vector<double> values;
   getValues(timevalue, &values);
   assert(values.size()==getParent()->getNBatch());
   if (outputstream!=NULL) {
      int nBatch = getParent()->getNBatch();
      int nk = getTargetLayer()->getNumGlobalNeurons();
      for (int b=0; b<nBatch; b++) {
         fprintf(outputstream->fp, "%st = %6.3f b = %d numNeurons = %8d %s = %f\n",
               getMessage(), timevalue, b, nk, getNormDescription(), values[b]);
      }
      fflush(outputstream->fp);
   }
   return PV_SUCCESS;
}

}  // end namespace PV
