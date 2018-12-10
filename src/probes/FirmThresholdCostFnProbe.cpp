/*
 * FirmThresholdCostFnProbe.cpp
 *
 *  Created on: Aug 14, 2015
 *      Author: pschultz
 */

#include "FirmThresholdCostFnProbe.hpp"
#include "components/ANNActivityBuffer.hpp" // To get VThresh and VWidth from targetLayer if it's an ANNLayer
#include "components/ActivityComponent.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

FirmThresholdCostFnProbe::FirmThresholdCostFnProbe() : AbstractNormProbe() { initialize_base(); }

FirmThresholdCostFnProbe::FirmThresholdCostFnProbe(
      const char *name,
      PVParams *params,
      Communicator *comm)
      : AbstractNormProbe() {
   initialize_base();
   initialize(name, params, comm);
}

int FirmThresholdCostFnProbe::initialize_base() {
   VThresh = (float)0;
   VWidth  = (float)0;
   return PV_SUCCESS;
}

FirmThresholdCostFnProbe::~FirmThresholdCostFnProbe() {}

void FirmThresholdCostFnProbe::initialize(const char *name, PVParams *params, Communicator *comm) {
   AbstractNormProbe::initialize(name, params, comm);
}

int FirmThresholdCostFnProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = AbstractNormProbe::ioParamsFillGroup(ioFlag);
   ioParam_VThresh(ioFlag);
   ioParam_VWidth(ioFlag);
   return status;
}

// We do not warn if VThresh and VWidth are absent, because if they are, we
// try to get the values from the targetLayer.
void FirmThresholdCostFnProbe::ioParam_VThresh(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, name, "VThresh", &VThresh, VThresh /*default*/, false /*warnIfAbsent*/);
}

void FirmThresholdCostFnProbe::ioParam_VWidth(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, name, "VWidth", &VWidth, VWidth /*default*/, false /*warnIfAbsent*/);
}

int FirmThresholdCostFnProbe::setNormDescription() {
   return setNormDescriptionToString("Cost function");
}

Response::Status FirmThresholdCostFnProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = AbstractNormProbe::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *activityComponent = getTargetLayer()->getComponentByType<ActivityComponent>();
   pvAssert(activityComponent);
   ANNActivityBuffer *activityBuffer = activityComponent->getComponentByType<ANNActivityBuffer>();
   if (activityBuffer != nullptr) {
      if (!parameters()->present(getName(), "VThresh")) {
         VThresh = activityBuffer->getVThresh();
      }
      if (!parameters()->present(getName(), "VWidth")) {
         VWidth = activityBuffer->getVWidth();
      }
   }
   else {
      // Reread VThresh and VWidth commands, this time warning if they are not
      // absent.
      parameters()->ioParamValue(
            PARAMS_IO_READ, name, "VThresh", &VThresh, VThresh /*default*/, true /*warnIfAbsent*/);
      parameters()->ioParamValue(
            PARAMS_IO_READ, name, "VThresh", &VThresh, VThresh /*default*/, true /*warnIfAbsent*/);
   }
   return Response::SUCCESS;
}

double FirmThresholdCostFnProbe::getValueInternal(double timevalue, int index) {
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
   double VThreshPlusVWidth = VThresh + VWidth;
   double amax              = 0.5 * VThreshPlusVWidth;
   double a2                = 0.5 / VThreshPlusVWidth;
   auto *publisherComponent = getTargetLayer()->getComponentByType<PublisherComponent>();
   int const numExtended    = publisherComponent->getNumExtended();
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
            if (maskLayerData[kexMask] == 0) {
               continue;
            }
            int featureBase = kxy * nf;
            for (int f = 0; f < nf; f++) {
               int kex  = kIndexExtended(featureBase++, nx, ny, nf, lt, rt, dn, up);
               double a = (double)fabs(aBuffer[kex]);
               if (a >= VThreshPlusVWidth) {
                  sum += amax;
               }
               else {
                  sum += a * (1.0 - a2 * a);
               }
            }
         }
      }
      else {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
         for (int k = 0; k < getTargetLayer()->getNumNeurons(); k++) {
            int kex  = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
            double a = (double)fabs(aBuffer[kex]);
            if (a == 0) {
               continue;
            }
            int kexMask = kIndexExtended(k, nx, ny, nf, maskLt, maskRt, maskDn, maskUp);
            if (maskLayerData[kexMask] == 0) {
               continue;
            }
            if (a >= VThreshPlusVWidth) {
               sum += amax;
            }
            else {
               sum += a * (1.0 - a2 * a);
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
            double a = inRestricted * (double)fabs(aBuffer[extIndex]);
            if (a >= VThreshPlusVWidth) {
               sum += amax;
            }
            else {
               sum += a * (1.0 - a2 * a);
            }
         }
      }
      else {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
         for (int k = 0; k < getTargetLayer()->getNumNeurons(); k++) {
            int kex  = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
            double a = (double)fabs(aBuffer[kex]);
            if (a == 0) {
               continue;
            }
            if (a >= VThreshPlusVWidth) {
               sum += amax;
            }
            else {
               sum += a * (1.0 - a2 * a);
            }
         }
      }
   }

   return sum;
}

} // end namespace PV
