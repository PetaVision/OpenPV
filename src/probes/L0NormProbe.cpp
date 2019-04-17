/*
 * L0NormProbe.cpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#include "L0NormProbe.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

L0NormProbe::L0NormProbe() : AbstractNormProbe() { initialize_base(); }

L0NormProbe::L0NormProbe(const char *name, PVParams *params, Communicator const *comm)
      : AbstractNormProbe() {
   initialize_base();
   initialize(name, params, comm);
}

L0NormProbe::~L0NormProbe() {}

void L0NormProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   AbstractNormProbe::initialize(name, params, comm);
}

int L0NormProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = AbstractNormProbe::ioParamsFillGroup(ioFlag);
   ioParam_nnzThreshold(ioFlag);
   return status;
}

void L0NormProbe::ioParam_nnzThreshold(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "nnzThreshold", &nnzThreshold, (float)0);
}

double L0NormProbe::getValueInternal(double timevalue, int index) {
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
   int sum                  = 0;
   auto *publisherComponent = getTargetLayer()->getComponentByType<BasePublisherComponent>();
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
                  float val = fabsf(aBuffer[kex]);
                  sum += val > nnzThreshold;
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
               float val = fabsf(aBuffer[kex]);
               sum += val > nnzThreshold;
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
            float val = inRestricted * fabsf(aBuffer[extIndex]);
            sum += val > nnzThreshold;
         }
      }
      else {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
         for (int k = 0; k < getTargetLayer()->getNumNeurons(); k++) {
            int kex   = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
            float val = fabsf(aBuffer[kex]);
            sum += val > nnzThreshold;
         }
      }
   }

   return (double)sum;
}

int L0NormProbe::setNormDescription() { return setNormDescriptionToString("L0-norm"); }

} // end namespace PV
