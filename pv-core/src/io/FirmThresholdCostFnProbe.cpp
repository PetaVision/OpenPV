/*
 * FirmThresholdCostFnProbe.cpp
 *
 *  Created on: Aug 14, 2015
 *      Author: pschultz
 */

#include "FirmThresholdCostFnProbe.hpp"
#include "../columns/HyPerCol.hpp"

namespace PV {

FirmThresholdCostFnProbe::FirmThresholdCostFnProbe() : AbstractNormProbe() {
   initFirmThresholdCostFnProbe_base();
}

FirmThresholdCostFnProbe::FirmThresholdCostFnProbe(const char * probeName, HyPerCol * hc) : AbstractNormProbe()
{
   initFirmThresholdCostFnProbe_base();
   initFirmThresholdCostFnProbe(probeName, hc);
}

int FirmThresholdCostFnProbe::initFirmThresholdCostFnProbe_base() {
   VThresh = (pvpotentialdata_t) 0;
   VWidth = (pvpotentialdata_t) 0;
   return PV_SUCCESS;
}

FirmThresholdCostFnProbe::~FirmThresholdCostFnProbe() {
}

int FirmThresholdCostFnProbe::initFirmThresholdCostFnProbe(const char * probeName, HyPerCol * hc) {
   return initialize(probeName, hc);
}

int FirmThresholdCostFnProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = AbstractNormProbe::ioParamsFillGroup(ioFlag);
   ioParam_VThresh(ioFlag);
   ioParam_VWidth(ioFlag);
   return status;
}

void FirmThresholdCostFnProbe::ioParam_VThresh(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValue(ioFlag, name, "VThresh", &VThresh, VThresh/*default*/, true/*warnIfAbsent*/);
   // TODO: Consider this possibility: don't warn if absent, but check during communicate if targetLayer is an ANNLayer,
   // and use its VWidth and VThresh.
}

void FirmThresholdCostFnProbe::ioParam_VWidth(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "VWidth", &VWidth, VWidth/*default*/, true/*warnIfAbsent*/);
   // TODO: sanity checking on values of VThresh and VWidth.
}

double FirmThresholdCostFnProbe::getValueInternal(double timevalue, int index) {
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
   pvpotentialdata_t threshpluswidth = VThresh+VWidth;
   double a2 = 0.5f*VThresh/VWidth;
   double amax=0.5f*VWidth*VThresh;
   pvadata_t const * aBuffer = getTargetLayer()->getLayerData() + index * getTargetLayer()->getNumExtended();
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif // PV_USE_OPENMP_THREADS
   for (int k=0; k<getTargetLayer()->getNumNeurons(); k++) {      
      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      pvadata_t a = fabsf(aBuffer[kex]);
      if (a>=VThresh) { sum += amax; }
      else { sum += a*(VThresh - a2*a); }
   }
   return sum;
}

}  // end namespace PV
