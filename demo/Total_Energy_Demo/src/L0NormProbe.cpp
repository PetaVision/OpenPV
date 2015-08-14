/*
 * L0NormProbe.cpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#include "L0NormProbe.hpp"
#include <columns/HyPerCol.hpp>

namespace PV {

L0NormProbe::L0NormProbe() : AbstractNormProbe() {
   initL0NormProbe_base();
}

L0NormProbe::L0NormProbe(const char * probeName, HyPerCol * hc) : AbstractNormProbe()
{
   initL0NormProbe_base();
   initL0NormProbe(probeName, hc);
}

L0NormProbe::~L0NormProbe() {
}

int L0NormProbe::initL0NormProbe(const char * probeName, HyPerCol * hc) {
   return initialize(probeName, hc);
}

int L0NormProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = AbstractNormProbe::ioParamsFillGroup(ioFlag);
   ioParam_nnzThreshold(ioFlag);
   return status;
}

void L0NormProbe::ioParam_nnzThreshold(enum ParamsIOFlag ioFlag) {
    getParent()->ioParamValue(ioFlag, getName(), "nnzThreshold", &nnzThreshold, (pvadata_t) 0);
}

double L0NormProbe::getValueInternal(double timevalue, int index) {
   if (index < 0 || index >= getParent()->getNBatch()) { return PV_FAILURE; }
   PVLayerLoc const * loc = getTargetLayer()->getLayerLoc();
   int const nx = loc->nx;
   int const ny = loc->ny;
   int const nf = loc->nf;
   PVHalo const * halo = &loc->halo;
   int const lt = halo->lt;
   int const rt = halo->rt;
   int const dn = halo->dn;
   int const up = halo->up;
   int sum = 0;
   pvadata_t const * aBuffer = getTargetLayer()->getLayerData() + index * getTargetLayer()->getNumExtended();
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif // PV_USE_OPENMP_THREADS
   for (int k=0; k<getTargetLayer()->getNumNeurons(); k++) {      
      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      sum += aBuffer[kex]>nnzThreshold || aBuffer[kex]<nnzThreshold;
   }
   return (double) sum;
}

int L0NormProbe::outputState(double timevalue) {
   std::vector<double> values;
   getValues(timevalue, &values);
   if (outputstream!=NULL) {
      int nBatch = getParent()->getNBatch();   
      int nk = getTargetLayer()->getNumGlobalNeurons();
      for (int b=0; b<nBatch; b++) {
         fprintf(outputstream->fp, "%st = %6.3f b = %d numNeurons = %8d L0-norm          = %f\n",
               getMessage(), timevalue, b, nk, values.at(b));
      }
      fflush(outputstream->fp);
   }
   return PV_SUCCESS;
}

}  // end namespace PV
