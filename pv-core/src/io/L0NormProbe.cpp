/*
 * L0NormProbe.cpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#include "L0NormProbe.hpp"
#include "../columns/HyPerCol.hpp"

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
   return initAbstractNormProbe(probeName, hc);
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

   if (getMaskLayer()) {
      PVLayerLoc const * maskLoc = getMaskLayer()->getLayerLoc();
      PVHalo const * maskHalo = &maskLoc->halo;
      pvadata_t const * maskLayerData = getMaskLayer()->getLayerData() + index*getMaskLayer()->getNumExtended(); // Is there a DataStore method to return the part of the layer data for a given batch index?
      int const maskLt = maskHalo->lt;
      int const maskRt = maskHalo->rt;
      int const maskDn = maskHalo->dn;
      int const maskUp = maskHalo->up;
      if (maskHasSingleFeature()) {
         assert(getTargetLayer()->getNumNeurons()==nx*ny*nf);
         int nxy = nx*ny;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
         for (int kxy=0; kxy<nxy; kxy++) {
            int kexMask = kIndexExtended(kxy, nx, ny, 1, maskLt, maskRt, maskDn, maskUp);
            if (maskLayerData[kexMask]) {
               int featureBase = kxy*nf;
               for (int f=0; f<nf; f++) {
                  int kex = kIndexExtended(featureBase++, nx, ny, nf, lt, rt, dn, up);
                  pvadata_t val = aBuffer[kex];
                  sum += (aBuffer[kex]>nnzThreshold || aBuffer[kex]<nnzThreshold);
               }
            }
         }         
      }
      else {
         #ifdef PV_USE_OPENMP_THREADS
         #pragma omp parallel for
         #endif // PV_USE_OPENMP_THREADS
         for (int k=0; k<getTargetLayer()->getNumNeurons(); k++) {
            int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
            int kexMask = kIndexExtended(k, nx, ny, nf, maskLt, maskRt, maskDn, maskUp);
            if (maskLayerData[kexMask]) {
               pvadata_t val = aBuffer[kex];
               sum += aBuffer[kex]>nnzThreshold || aBuffer[kex]<nnzThreshold;
            }
         }
      }
   }
   else {
      #ifdef PV_USE_OPENMP_THREADS
      #pragma omp parallel for
      #endif // PV_USE_OPENMP_THREADS
      for (int k=0; k<getTargetLayer()->getNumNeurons(); k++) {      
         int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         pvadata_t val = aBuffer[kex];
         sum += aBuffer[kex]>nnzThreshold || aBuffer[kex]<nnzThreshold;
      }
   }
   
   return (double) sum;
}

int L0NormProbe::setNormDescription() {
   return setNormDescriptionToString("L0-norm");
}

}  // end namespace PV
