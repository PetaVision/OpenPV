/*
 * L2NormProbe.cpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#include "L2NormProbe.hpp"
#include "../columns/HyPerCol.hpp"

namespace PV {

L2NormProbe::L2NormProbe() : AbstractNormProbe() {
   initL2NormProbe_base();
}

L2NormProbe::L2NormProbe(const char * probeName, HyPerCol * hc) : AbstractNormProbe()
{
   initL2NormProbe_base();
   initL2NormProbe(probeName, hc);
}

L2NormProbe::~L2NormProbe() {
}

int L2NormProbe::initL2NormProbe(const char * probeName, HyPerCol * hc) {
   return initialize(probeName, hc);
}

int L2NormProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_exponent(ioFlag);
   return status;
}

void L2NormProbe::ioParam_exponent(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "energyProbe"));
   if (energyProbe && energyProbe[0]) {
      parent->ioParamValue(ioFlag, name, "exponent", &exponent, 1.0/*default*/, true/*warnIfAbsent*/);
   }
}

double L2NormProbe::getValueInternal(double timevalue, int index) {
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
   double l2normsq = 0.0;
   pvadata_t const * aBuffer = getTargetLayer()->getLayerData() + index * getTargetLayer()->getNumExtended();
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif // PV_USE_OPENMP_THREADS
   for (int k=0; k<getTargetLayer()->getNumNeurons(); k++) {      
      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      pvadata_t val = aBuffer[kex];
      l2normsq += fabs(val*val);
   }
   return l2normsq;
}

int L2NormProbe::getValues(double timevalue, std::vector<double> * values) {
   int status = AbstractNormProbe::getValues(timevalue, values);
   if (status != PV_SUCCESS) { return status; }
   if (exponent != 2.0) {
      int nBatch = getParent()->getNBatch();
      for (int b=0; b<nBatch; b++) {
         double * vptr = &(*values)[b];
         *vptr = pow(*vptr, exponent/2.0);
      }
   }
   return PV_SUCCESS;
}
   
double L2NormProbe::getValue(double timevalue, int index) {
   double v = AbstractNormProbe::getValue(timevalue, index);
   if (exponent != 2.0) { v = pow(v, exponent/2.0); }
   return v;
}

int L2NormProbe::outputState(double timevalue) {
   std::vector<double> values;
   getValues(timevalue, &values);
   if (outputstream!=NULL) {
      int nBatch = getParent()->getNBatch();
      int nk = getTargetLayer()->getNumGlobalNeurons();
      if (exponent==1.0) {
         for (int b=0; b<nBatch; b++) {
            fprintf(outputstream->fp, "%st = %6.3f b = %d numNeurons = %8d, L2-norm          = %f\n",
                  getMessage(), timevalue, b, nk, values.at(b));
         }
      }
      else if (exponent==2.0) {
         for (int b=0; b<nBatch; b++) {
            fprintf(outputstream->fp, "%st = %6.3f b = %d numNeurons = %8d, L2-norm squared  = %f\n",
                  getMessage(), timevalue, b, nk, values.at(b));
         }
      }
      else {
         for (int b=0; b<nBatch; b++) {
            fprintf(outputstream->fp, "%st = %6.3f b = %d numNeurons = %8d, L2-norm^%f = %f\n",
                  getMessage(), timevalue, b, nk, exponent, values.at(b));
         }
      }
      fflush(outputstream->fp);
   }
   return PV_SUCCESS;
}

}  // end namespace PV
