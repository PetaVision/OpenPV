/*
 * L1NormProbe.cpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#include <cmath>

#include "L1NormProbe.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

L1NormProbe::L1NormProbe() : AbstractNormProbe() { initialize_base(); }

L1NormProbe::L1NormProbe(const char *name, PVParams *params, Communicator *comm)
      : AbstractNormProbe() {
   initialize_base();
   initialize(name, params, comm);
}

L1NormProbe::~L1NormProbe() {}

void L1NormProbe::initialize(const char *name, PVParams *params, Communicator *comm) {
   AbstractNormProbe::initialize(name, params, comm);
}

double L1NormProbe::getValueInternal(double timevalue, int index) {
   PVLayerLoc const *loc = getTargetLayer()->getLayerLoc();
   if (index < 0 || index >= loc->nbatch) {
      return PV_FAILURE;
   }
   int const nx             = loc->nx;
   int const ny             = loc->ny;
   int const nf             = loc->nf;
   PVHalo const *halo       = &loc->halo;
   int const lt             = halo->lt;
   int const rt             = halo->rt;
   int const dn             = halo->dn;
   int const up             = halo->up;
   double sum               = 0.0;
   auto *publisherComponent = getTargetLayer()->getComponentByType<PublisherComponent>();
   int const numExtended    = (nx + lt + rt) * (ny + dn + up) * nf;
   float const *aBuffer     = publisherComponent->getLayerData() + index * numExtended;

   if (getMaskLayerData()) {
      PVLayerLoc const *maskLoc  = getMaskLayerData()->getLayerLoc();
      int const maskLt           = maskLoc->halo.lt;
      int const maskRt           = maskLoc->halo.rt;
      int const maskDn           = maskLoc->halo.dn;
      int const maskUp           = maskLoc->halo.up;
      int const maskNumExtended  = getMaskLayerData()->getNumExtended();
      float const *maskLayerData = getMaskLayerData()->getLayerData() + index * maskNumExtended;
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
      if (publisherComponent->getSparseLayer()) {
         PVLayerCube cube   = publisherComponent->getPublisher()->createCube();
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
