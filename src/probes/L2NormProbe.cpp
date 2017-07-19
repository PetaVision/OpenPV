/*
 * L2NormProbe.cpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#include "L2NormProbe.hpp"
#include "columns/HyPerCol.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

L2NormProbe::L2NormProbe() : AbstractNormProbe() { initialize_base(); }

L2NormProbe::L2NormProbe(const char *name, HyPerCol *hc) : AbstractNormProbe() {
   initialize_base();
   initialize(name, hc);
}

L2NormProbe::~L2NormProbe() {}

int L2NormProbe::initialize_base() {
   exponent = 1.0;
   return PV_SUCCESS;
}

int L2NormProbe::initialize(const char *name, HyPerCol *hc) {
   return AbstractNormProbe::initialize(name, hc);
}

int L2NormProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = AbstractNormProbe::ioParamsFillGroup(ioFlag);
   ioParam_exponent(ioFlag);
   return status;
}

void L2NormProbe::ioParam_exponent(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "exponent", &exponent, 1.0 /*default*/, true /*warnIfAbsent*/);
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
   double l2normsq       = 0.0;
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
      if (getTargetLayer()->getSparseFlag()) {
         PVLayerCube cube   = getTargetLayer()->getPublisher()->createCube();
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

int L2NormProbe::calcValues(double timevalue) {
   int status = AbstractNormProbe::calcValues(timevalue);
   if (status != PV_SUCCESS) {
      return status;
   }
   if (exponent != 2.0) {
      double *valBuf = getValuesBuffer();
      int numVals    = this->getNumValues();
      for (int b = 0; b < numVals; b++) {
         double v  = valBuf[b];
         valBuf[b] = pow(v, exponent / 2.0);
      }
   }
   return PV_SUCCESS;
}

} // end namespace PV
