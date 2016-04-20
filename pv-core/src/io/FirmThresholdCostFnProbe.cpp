/*
 * FirmThresholdCostFnProbe.cpp
 *
 *  Created on: Aug 14, 2015
 *      Author: pschultz
 */

#include "FirmThresholdCostFnProbe.hpp"
#include "../columns/HyPerCol.hpp"
#include "../layers/ANNLayer.hpp" // To get VThresh and VWidth from targetLayer if it's an ANNLayer

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
   return initAbstractNormProbe(probeName, hc);
}

int FirmThresholdCostFnProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = AbstractNormProbe::ioParamsFillGroup(ioFlag);
   ioParam_VThresh(ioFlag);
   ioParam_VWidth(ioFlag);
   return status;
}

void FirmThresholdCostFnProbe::ioParam_VThresh(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValue(ioFlag, name, "VThresh", &VThresh, VThresh/*default*/, false/*warnIfAbsent*/);
}

void FirmThresholdCostFnProbe::ioParam_VWidth(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "VWidth", &VWidth, VWidth/*default*/, false/*warnIfAbsent*/);
}

int FirmThresholdCostFnProbe::setNormDescription() {
   return setNormDescriptionToString("Cost function");
}

int FirmThresholdCostFnProbe::communicateInitInfo() {
   AbstractNormProbe::communicateInitInfo();
   ANNLayer * targetANNLayer = dynamic_cast<ANNLayer *>(getTargetLayer());
   if (targetANNLayer!=NULL) {
      if (!getParent()->parameters()->present(getName(), "VThresh")) {
         VThresh=targetANNLayer->getVThresh();
      }
      if (!getParent()->parameters()->present(getName(), "VWidth")) {
         VWidth=targetANNLayer->getVWidth();
      }
   }
   else {
      // Reread VThresh and VWidth commands, this time warning if they are not absent.
      parent->ioParamValue(PARAMS_IO_READ, name, "VThresh", &VThresh, VThresh/*default*/, true/*warnIfAbsent*/);
      parent->ioParamValue(PARAMS_IO_READ, name, "VThresh", &VThresh, VThresh/*default*/, true/*warnIfAbsent*/);
   }
   return PV_SUCCESS;
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
   pvpotentialdata_t VThreshPlusVWidth = VThresh+VWidth;
   double amax=0.5f*VThreshPlusVWidth;
   double a2 = 0.5f/VThreshPlusVWidth;
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
            if (maskLayerData[kexMask]==0) { continue; }
            int featureBase = kxy*nf;
            for (int f=0; f<nf; f++) {
               int kex = kIndexExtended(featureBase++, nx, ny, nf, lt, rt, dn, up);
               pvadata_t a = fabsf(aBuffer[kex]);
               if (a>=VThreshPlusVWidth) {
                  sum += amax;
               }
               else {
                  sum += a*(1 - a2*a);
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
            pvadata_t a = fabsf(aBuffer[kex]);
            if (a==0) { continue; }
            int kexMask = kIndexExtended(k, nx, ny, nf, maskLt, maskRt, maskDn, maskUp);
            if (maskLayerData[kexMask]==0) { continue; }
            if (a>=VThreshPlusVWidth) {
               sum += amax;
            }
            else {
               sum += a*(1 - a2*a);
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
            pvadata_t a = inRestricted * fabsf(aBuffer[extIndex]);
            if (a>=VThreshPlusVWidth) {
               sum += amax;
            }
            else {
               sum += a*(1 - a2*a);
            }
         }
      }
      else {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif // PV_USE_OPENMP_THREADS
         for (int k=0; k<getTargetLayer()->getNumNeurons(); k++) {
            int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
            pvadata_t a = fabsf(aBuffer[kex]);
            if (a==0) { continue; }
            if (a>=VThreshPlusVWidth) {
               sum += amax;
            }
            else {
               sum += a*(1 - a2*a);
            }
         }
      }
   }
   
   return sum;
}

BaseObject * createFirmThresholdCostFnProbe(char const * name, HyPerCol * hc) {
   return hc ? new FirmThresholdCostFnProbe(name, hc) : NULL;
}

}  // end namespace PV
