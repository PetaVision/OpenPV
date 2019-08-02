/*
 * L2NormProbe.cpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#include "L2NormProbe.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

L2NormProbe::L2NormProbe() : AbstractNormProbe() { initialize_base(); }

L2NormProbe::L2NormProbe(const char *name, PVParams *params, Communicator const *comm)
      : AbstractNormProbe() {
   initialize_base();
   initialize(name, params, comm);
}

L2NormProbe::~L2NormProbe() {}

int L2NormProbe::initialize_base() {
   exponent = 1.0;
   return PV_SUCCESS;
}

void L2NormProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   AbstractNormProbe::initialize(name, params, comm);
}

int L2NormProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = AbstractNormProbe::ioParamsFillGroup(ioFlag);
   ioParam_exponent(ioFlag);
   return status;
}

void L2NormProbe::ioParam_exponent(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, name, "exponent", &exponent, 1.0 /*default*/, true /*warnIfAbsent*/);
}

int L2NormProbe::setNormDescription() {
   assert(!parameters()->presentAndNotBeenRead(name, "exponent"));
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
   double l2normsq          = 0.0;
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
#pragma omp parallel for reduction(+ : l2normsq)
#endif // PV_USE_OPENMP_THREADS
         for (int kxy = 0; kxy < nxy; kxy++) {
            int kexMask = kIndexExtended(kxy, nx, ny, 1, maskLt, maskRt, maskDn, maskUp);
            if (maskLayerData[kexMask]) {
               int featureBase = kxy * nf;
               for (int f = 0; f < nf; f++) {
                  int kex = kIndexExtended(featureBase++, nx, ny, nf, lt, rt, dn, up);
                  l2normsq += pow((double)aBuffer[kex], 2.0);
               }
            }
         }
      }
      else {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : l2normsq)
#endif // PV_USE_OPENMP_THREADS
         for (int k = 0; k < getTargetLayer()->getNumNeurons(); k++) {
            int kex     = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
            int kexMask = kIndexExtended(k, nx, ny, nf, maskLt, maskRt, maskDn, maskUp);
            double val  = aBuffer[kex];
            l2normsq += (double)maskLayerData[kexMask] * val * val;
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
#pragma omp parallel for reduction(+ : l2normsq)
#endif // PV_USE_OPENMP_THREADS
         for (int k = 0; k < numActive; k++) {
            int extIndex     = activeList[k].index;
            int inRestricted = !extendedIndexInBorderRegion(
                  extIndex, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
            double val = inRestricted * fabs((double)aBuffer[extIndex]);
            l2normsq += pow(val, 2.0);
         }
      }
      else {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : l2normsq)
#endif // PV_USE_OPENMP_THREADS
         for (int k = 0; k < getTargetLayer()->getNumNeurons(); k++) {
            int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
            l2normsq += pow((double)aBuffer[kex], 2.0);
         }
      }
   }

   return l2normsq;
}

void L2NormProbe::calcValues(double timevalue) {
   AbstractNormProbe::calcValues(timevalue);
   if (exponent != 2.0) {
      double *valBuf = getValuesBuffer();
      int numVals    = this->getNumValues();
      for (int b = 0; b < numVals; b++) {
         double v  = valBuf[b];
         valBuf[b] = pow(v, exponent / 2.0);
      }
   }
}

} // end namespace PV
