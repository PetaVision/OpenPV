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
   free(maskLayerName); maskLayerName = NULL;
   // Don't free maskLayer, which belongs to the HyPerCol.
}

int AbstractNormProbe::initAbstractNormProbe_base() {
   normDescription = NULL;
   maskLayerName = NULL;
   maskLayer = NULL;
   singleFeatureMask = false;
   timeLastComputed = -std::numeric_limits<double>::infinity();
   return PV_SUCCESS;
}

int AbstractNormProbe::initAbstractNormProbe(const char * probeName, HyPerCol * hc) {
   int status = LayerProbe::initialize(probeName, hc);
   if (status == PV_SUCCESS) {
      status = setNormDescription();
   }
   return status;
}
   
int AbstractNormProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_maskLayerName(ioFlag);
   return status;
}
   
void AbstractNormProbe::ioParam_maskLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "maskLayerName", &maskLayerName, NULL, false/*warnIfAbsent*/);
}

int AbstractNormProbe::communicateInitInfo() {
   int status = LayerProbe::communicateInitInfo();
   assert(targetLayer);
   if (maskLayerName && maskLayerName[0]) {
      maskLayer = parent->getLayerFromName(maskLayerName);
      if (maskLayer==NULL) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: maskLayerName \"%s\" is not a layer in the HyPerCol.\n",
                    this->getKeyword(), name, maskLayerName);
         }
#ifdef PV_USE_MPI
         MPI_Barrier(parent->icCommunicator()->communicator());
#endif
         exit(EXIT_FAILURE);
      }

      const PVLayerLoc * maskLoc = maskLayer->getLayerLoc();
      const PVLayerLoc * loc = targetLayer->getLayerLoc();
      assert(maskLoc != NULL && loc != NULL);
      if (maskLoc->nxGlobal != loc->nxGlobal || maskLoc->nyGlobal != loc->nyGlobal) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: maskLayerName \"%s\" does not have the same x and y dimensions.\n",
                    this->getKeyword(), name, maskLayerName);
            fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                    maskLoc->nxGlobal, maskLoc->nyGlobal, maskLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
         }
#ifdef PV_USE_MPI
         MPI_Barrier(parent->icCommunicator()->communicator());
#endif
         exit(EXIT_FAILURE);
      }

      if(maskLoc->nf != 1 && maskLoc->nf != loc->nf){
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: maskLayerName \"%s\" must either have the same number of features as this layer, or one feature.\n",
                    this->getKeyword(), name, maskLayerName);
            fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                    maskLoc->nxGlobal, maskLoc->nyGlobal, maskLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
         }
#ifdef PV_USE_MPI
         MPI_Barrier(parent->icCommunicator()->communicator());
#endif
         exit(EXIT_FAILURE);
      }
      assert(maskLoc->nx==loc->nx && maskLoc->ny==loc->ny);
      singleFeatureMask = maskLoc->nf==1 && loc->nf !=1;
   }
   return status;
}

int AbstractNormProbe::setNormDescription() {
    return setNormDescriptionToString("norm");
}

int AbstractNormProbe::setNormDescriptionToString(char const * s) {
    normDescription = strdup(s);
    return normDescription ? PV_SUCCESS : PV_FAILURE;
}

int AbstractNormProbe::calcValues(double timeValue) {
   double * valuesBuffer = this->getValuesBuffer();
   for (int b=0; b<this->getNumValues(); b++) {
      valuesBuffer[b] = getValueInternal(timeValue, b);
   }
   MPI_Allreduce(MPI_IN_PLACE, valuesBuffer, getNumValues(), MPI_DOUBLE, MPI_SUM, parent->icCommunicator()->communicator());
   return PV_SUCCESS;
}

int AbstractNormProbe::outputState(double timevalue) {
   getValues(timevalue);
   double * valuesBuffer = this->getValuesBuffer();
   if (outputstream!=NULL) {
      int nBatch = getNumValues();
      int nk = getTargetLayer()->getNumGlobalNeurons();
      for (int b=0; b<nBatch; b++) {
         fprintf(outputstream->fp, "%st = %6.3f b = %d numNeurons = %8d %s = %f\n",
               getMessage(), timevalue, b, nk, getNormDescription(), valuesBuffer[b]);
      }
      fflush(outputstream->fp);
   }
   return PV_SUCCESS;
}

}  // end namespace PV
