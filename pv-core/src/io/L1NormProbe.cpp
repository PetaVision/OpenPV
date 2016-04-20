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
   return initAbstractNormProbe(probeName, hc);
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
                  sum += fabs(val);
               }
            }
         }         
      }
      else {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
         for (int k=0; k<getTargetLayer()->getNumNeurons(); k++) {
            int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
            int kexMask = kIndexExtended(k, nx, ny, nf, maskLt, maskRt, maskDn, maskUp);
            if (maskLayerData[kexMask]) {
               pvadata_t val = aBuffer[kex];
               sum += fabs(val);
            }
         }
      }
   }
   else {
      if (getTargetLayer()->getSparseFlag()) {
         DataStore * store = parent->icCommunicator()->publisherStore(getTargetLayer()->getLayerId());
         int numActive = (int) store->numActiveBuffer(index)[0];
         unsigned int const * activeList = store->activeIndicesBuffer(index);
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
         for (int k=0; k<numActive; k++) {
            int extIndex = activeList[k];
            int inRestricted = !extendedIndexInBorderRegion(extIndex, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
            pvadata_t val = inRestricted * fabsf(aBuffer[extIndex]);
            sum += fabsf(val);
         }
      }
      else {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
         for (int k=0; k<getTargetLayer()->getNumNeurons(); k++) {
            int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
            pvadata_t val = fabsf(aBuffer[kex]);
            sum += fabsf(val);
         }
      }
   }
   
   return sum;
}

int L1NormProbe::setNormDescription() {
   return setNormDescriptionToString("L1-norm");
}

BaseObject * createL1NormProbe(char const * name, HyPerCol * hc) {
   return hc ? new L1NormProbe(name, hc) : NULL;
}

}  // end namespace PV
