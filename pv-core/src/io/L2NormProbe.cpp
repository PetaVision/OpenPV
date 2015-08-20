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

int L2NormProbe::initL2NormProbe_base() {
   exponent = 1.0;
   return PV_SUCCESS;
}

int L2NormProbe::initL2NormProbe(const char * probeName, HyPerCol * hc) {
   return initAbstractNormProbe(probeName, hc);
}

int L2NormProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_exponent(ioFlag);
   return status;
}

void L2NormProbe::ioParam_exponent(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "exponent", &exponent, 1.0/*default*/, true/*warnIfAbsent*/);
}

int L2NormProbe::setNormDescription() {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "exponent"));
   int status = PV_SUCCESS;
   if (exponent == 1.0) {
      status = setNormDescriptionToString("L2-norm");
   }
   else if (exponent == 2.0) {
      status = setNormDescriptionToString("L2-norm squared");
   }
   else {
      std::stringstream desc("(L2-norm)^");
      desc << exponent;
      status = setNormDescriptionToString(desc.str().c_str());
   };
   return status;
};

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

}  // end namespace PV
