/*
 * LayerFunctionProbe.cpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#include "LayerFunctionProbe.hpp"

namespace PV {

LayerFunctionProbe::LayerFunctionProbe(const char * probeName, HyPerCol * hc)
   : StatsProbe()
{
   initLayerFunctionProbe_base();
   initLayerFunctionProbe(probeName, hc);
}

LayerFunctionProbe::LayerFunctionProbe()
   : StatsProbe()
{
   initLayerFunctionProbe_base();
   // Derived classes should call LayerFunctionProbe::initLayerFunctionProbe
}

LayerFunctionProbe::~LayerFunctionProbe() {
   free(parentGenColProbeName); parentGenColProbeName = NULL;
   delete function;
}

int LayerFunctionProbe::initLayerFunctionProbe_base() {
   function = NULL;
   parentGenColProbeName = NULL;
   return PV_SUCCESS;
}

int LayerFunctionProbe::initLayerFunctionProbe(const char * probeName, HyPerCol * hc) {
   int status = initStatsProbe(probeName, hc);
   if (status == PV_SUCCESS) {
      initFunction();
      if (function==NULL) {
         fprintf(stderr, "%s \"%s\" error: rank %d unable to construct LayerFunction.\n",
               getParent()->parameters()->groupKeywordFromName(probeName), probeName, getParent()->columnId());
         status = PV_FAILURE;
         exit(EXIT_FAILURE);
      }
   }
   return status;
}

void LayerFunctionProbe::initFunction() {
   function = new LayerFunction(getName());
}

int LayerFunctionProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbe::ioParamsFillGroup(ioFlag);
   ioParam_parentGenColProbe(ioFlag);
   ioParam_coeff(ioFlag);
   return status;
}

void LayerFunctionProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      requireType(BufV);
   }
}

void LayerFunctionProbe::ioParam_parentGenColProbe(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamString(ioFlag, getName(), "parentGenColProbe", &parentGenColProbeName, NULL, false/*warnIfAbsent*/);
}

void LayerFunctionProbe::ioParam_coeff(enum ParamsIOFlag ioFlag) {
   assert(!getParent()->parameters()->presentAndNotBeenRead(getName(), "parentGenColProbeName"));
   if (parentGenColProbeName != NULL && parentGenColProbeName[0] != '\0') {
      getParent()->ioParamValue(ioFlag, getName(), "coeff", &coeff, (pvdata_t) 1.0);
   }
}

int LayerFunctionProbe::communicateInitInfo() {
   int status = StatsProbe::communicateInitInfo();
   if (status == PV_SUCCESS && parentGenColProbeName != NULL) {
      if (parentGenColProbeName != NULL && parentGenColProbeName[0] != '\0') {
         ColProbe * colprobe = getParent()->getColProbeFromName(parentGenColProbeName);
         GenColProbe * gencolprobe = dynamic_cast<GenColProbe *>(colprobe);
         if (gencolprobe==NULL) {
            if (getParent()->columnId()==0) {
               fprintf(stderr, "%s \"%s\" error: parentGenColProbe \"%s\" is not a GenColProbe in the column.\n",
                     getParent()->parameters()->groupKeywordFromName(getName()), getName(), parentGenColProbeName);
            }
#ifdef PV_USE_MPI
            MPI_Barrier(getParent()->icCommunicator()->communicator());
#endif
            exit(EXIT_FAILURE);
         }
         status = gencolprobe->addLayerTerm((LayerFunctionProbe *) this, getTargetLayer(), coeff);
      }
   }
   return status;
}

int LayerFunctionProbe::outputState(double timef) {
   for(int b = 0; b < getTargetLayer()->getParent()->getNBatch(); b++){
      pvdata_t val = function->evaluate(timef, getTargetLayer(), b);
#ifdef PV_USE_MPI
      if( getTargetLayer()->getParent()->icCommunicator()->commRank() != 0 ) return PV_SUCCESS;
#endif // PV_USE_MPI
      if( function ) {
         return writeState(timef, getTargetLayer(), b, val);
      }
      else {
         fprintf(stderr, "LayerFunctionProbe \"%s\" for layer %s: function has not been set\n", getMessage(), getTargetLayer()->getName());
         return PV_FAILURE;
      }
   }
   return PV_SUCCESS;
}  // end LayerFunctionProbe::outputState(float, HyPerLayer *)

int LayerFunctionProbe::writeState(double timef, HyPerLayer * l, int batchIdx, pvdata_t value) {
#ifdef PV_USE_MPI
   // In MPI mode, this function should only be called by the root processor.
   assert(l->getParent()->icCommunicator()->commRank() == 0);
#endif // PV_USE_MPI
   int printstatus = fprintf(outputstream->fp, "%st = %6.3f b = %d numNeurons = %8d Value            = %f\n", getMessage(), timef, batchIdx, l->getNumGlobalNeurons(), value);
   return printstatus > 0 ? PV_SUCCESS : PV_FAILURE;
}

}  // end namespace PV
