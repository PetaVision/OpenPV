/*
 * L1NormProbe.cpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#include <cmath>

#include "L1NormProbe.hpp"
#include "columns/HyPerCol.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

L1NormProbe::L1NormProbe() : AbstractNormProbe() { initialize_base(); }

L1NormProbe::L1NormProbe(const char *name, HyPerCol *hc) : AbstractNormProbe() {
   initialize_base();
   initialize(name, hc);
}

L1NormProbe::~L1NormProbe() {}

int L1NormProbe::initialize(const char *name, HyPerCol *hc) {
   return AbstractNormProbe::initialize(name, hc);
}

double L1NormProbe::getValueInternal(double timevalue, int index) {
   if (index < 0 || index >= parent->getNBatch()) {
      return PV_FAILURE;
   }
   PVLayerLoc const *loc = getTargetLayer()->getLayerLoc();
   int const nx          = loc->nx;
   int const ny          = loc->ny;
   int const nf          = loc->nf;
   PVHalo const *halo    = &loc->halo;
   int const lt          = halo->lt;
   int const rt          = halo->rt;
   int const dn          = halo->dn;
   int const up          = halo->up;
   double sum            = 0.0;
   float const *aBuffer =
         getTargetLayer()->getLayerData() + index * getTargetLayer()->getNumExtended();
   if (getMaskLayer()) {
      PVLayerLoc const *maskLoc = getMaskLayer()->getLayerLoc();
      PVHalo const *maskHalo    = &maskLoc->halo;
      float const *maskLayerData =
            getMaskLayer()->getLayerData()
            + index * getMaskLayer()->getNumExtended(); // Is there a DataStore method to return the
      // part of the layer data for a given batch
      // index?
      int const maskLt = maskHalo->lt;
      int const maskRt = maskHalo->rt;
      int const maskDn = maskHalo->dn;
      int const maskUp = maskHalo->up;
      if (maskHasSingleFeature()) {
         assert(getTargetLayer()->getNumNeurons() == nx * ny * nf);
         int nxy = nx * ny;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
         for (int kxy = 0; kxy < nxy; kxy++) {
            int kexMask = kIndexExtended(kxy, nx, ny, 1, maskLt, maskRt, maskDn, maskUp);
            if (maskLayerData[kexMask]) {
               int featureBase = kxy * nf;
               for (int f = 0; f < nf; f++) {
                  int kex   = kIndexExtended(featureBase++, nx, ny, nf, lt, rt, dn, up);
                  float val = aBuffer[kex];
                  sum += (double)std::fabs(val);
               }
            }
         }
      }
      else {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
         for (int k = 0; k < getTargetLayer()->getNumNeurons(); k++) {
            int kex     = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
            int kexMask = kIndexExtended(k, nx, ny, nf, maskLt, maskRt, maskDn, maskUp);
            if (maskLayerData[kexMask]) {
               float val = aBuffer[kex];
               sum += std::fabs((double)val);
            }
         }
      }
   }
   else {
      if (getTargetLayer()->getSparseFlag()) {
         PVLayerCube cube   = getTargetLayer()->getPublisher()->createCube();
         long int numActive = cube.numActive[index];
         int numItems       = cube.numItems / cube.loc.nbatch;
         SparseList<float>::Entry const *activeList =
               &((SparseList<float>::Entry *)cube.activeIndices)[index * numItems];
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
         for (int k = 0; k < numActive; k++) {
            int extIndex     = activeList[k].index;
            int inRestricted = !extendedIndexInBorderRegion(
                  extIndex, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
            double val = inRestricted * (double)fabs(aBuffer[extIndex]);
            sum += fabs(val);
         }
      }
      else {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
         for (int k = 0; k < getTargetLayer()->getNumNeurons(); k++) {
            int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
            sum += (double)fabs(aBuffer[kex]);
         }
      }
   }

   return sum;
}

int L1NormProbe::setNormDescription() { return setNormDescriptionToString("L1-norm"); }

} // end namespace PV
