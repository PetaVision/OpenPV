/*
 * L1NormProbe.cpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#include "L1NormProbe.hpp"
#include "../columns/HyPerCol.hpp"

namespace PV {

L1NormProbe::L1NormProbe() : AbstractNormProbe() {
   initL1NormProbe_base();
}

L1NormProbe::L1NormProbe(const char * probeName, HyPerCol * hc) : AbstractNormProbe()
{
   initL1NormProbe_base();
   initL1NormProbe(probeName, hc);
}

L1NormProbe::~L1NormProbe() {
}

int L1NormProbe::initL1NormProbe(const char * probeName, HyPerCol * hc) {
   return initialize(probeName, hc);
}

double L1NormProbe::getValueInternal(double timevalue, int index) {
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
   double sum = 0.0;
   pvadata_t const * aBuffer = getTargetLayer()->getLayerData() + index * getTargetLayer()->getNumExtended();
   for (int k=0; k<getTargetLayer()->getNumNeurons(); k++) {      
      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      sum += fabs(aBuffer[kex]);
   }
   return sum;
}

int L1NormProbe::outputState(double timevalue) {
   std::vector<double> values;
   getValues(timevalue, &values);
   if (outputstream!=NULL) {
      int nBatch = getParent()->getNBatch();   
      int nk = getTargetLayer()->getNumGlobalNeurons();
      for (int b=0; b<nBatch; b++) {
         fprintf(outputstream->fp, "%st = %6.3f b = %d numNeurons = %8d L1-norm          = %f\n",
               getMessage(), timevalue, b, nk, values.at(b));
      }
      fflush(outputstream->fp);
   }
   return PV_SUCCESS;
}

}  // end namespace PV
