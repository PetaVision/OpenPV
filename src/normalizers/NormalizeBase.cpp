/*
 * NormalizeBase.cpp
 *
 *  Created on: Apr 5, 2013
 *      Author: pschultz
 */

#include "NormalizeBase.hpp"

namespace PV {

NormalizeBase::NormalizeBase() { initialize_base(); }

NormalizeBase::~NormalizeBase() {
   free(connectionList); // normalizer does not own the individual connections in the list, so don't
   // free them
}

int NormalizeBase::initialize_base() {
   connectionList              = NULL;
   numConnections              = 0;
   strength                    = 1.0f;
   normalizeArborsIndividually = false;
   normalizeOnInitialize       = true;
   normalizeOnWeightUpdate     = true;
   return PV_SUCCESS;
}

// NormalizeBase does not directly call initialize since it is an abstract base class.
// Subclasses should call NormalizeBase::initialize from their own initialize routine
// This allows virtual methods called from initialize to be aware of which class's constructor was
// called.
int NormalizeBase::initialize(const char *name, HyPerCol *hc) {
   // name is the name of a group in the PVParams object.  Parameters related to normalization
   // should be in the indicated group.

   int status           = BaseObject::initialize(name, hc);
   this->connectionList = NULL;
   this->numConnections = 0;
   status               = hc->addNormalizer(this);
   return status;
}

int NormalizeBase::setDescription() {
   description.clear();
   char const *method = parent->parameters()->stringValue(
         name, "normalizeMethod", false /*do not warn if absent*/);
   if (method == nullptr) {
      description.append("weight normalizer ");
   }
   else {
      description.append(method);
   }
   description.append(" \"").append(name).append("\"");
   return PV_SUCCESS;
}

int NormalizeBase::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_strength(ioFlag);
   ioParam_normalizeArborsIndividually(ioFlag);
   ioParam_normalizeOnInitialize(ioFlag);
   ioParam_normalizeOnWeightUpdate(ioFlag);
   return PV_SUCCESS;
}

void NormalizeBase::ioParam_strength(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "strength", &strength, strength /*default*/, true /*warn if absent*/);
}

void NormalizeBase::ioParam_normalizeArborsIndividually(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "normalizeArborsIndividually",
         &normalizeArborsIndividually,
         false /*default*/,
         true /*warnIfAbsent*/);
}

void NormalizeBase::ioParam_normalizeOnInitialize(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "normalizeOnInitialize", &normalizeOnInitialize, normalizeOnInitialize);
}

void NormalizeBase::ioParam_normalizeOnWeightUpdate(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "normalizeOnWeightUpdate",
         &normalizeOnWeightUpdate,
         normalizeOnWeightUpdate);
}

int NormalizeBase::communicateInitInfo() { return addConnToList(getTargetConn()); }

HyPerConn *NormalizeBase::getTargetConn() {
   BaseConnection *baseConn = parent->getConnFromName(name);
   HyPerConn *targetConn    = dynamic_cast<HyPerConn *>(baseConn);
   pvAssertMessage(
         targetConn, "%s: target connection \"%s\" is not a HyPerConn\n", getDescription_c(), name);
   return targetConn;
}

int NormalizeBase::normalizeWeightsWrapper() {
   int status      = PV_SUCCESS;
   double simTime  = parent->simulationTime();
   bool needUpdate = false;
   if (normalizeOnInitialize && simTime == parent->getStartTime()) {
      needUpdate = true;
   }
   else if (!normalizeOnWeightUpdate) {
      needUpdate = false;
   }
   else {
      for (int k = 0; k < this->numConnections; k++) {
         HyPerConn *callingConn = connectionList[k];
         if (simTime == callingConn->getLastUpdateTime()) {
            needUpdate = true;
         }
      }
   }
   if (needUpdate) {
      status = normalizeWeights();
   }
   // Need to set each connection's last update time to simTime
   return status;
}

int NormalizeBase::normalizeWeights() {
   int status = PV_SUCCESS;
   for (int c = 0; c < numConnections; c++) {
      HyPerConn *conn = connectionList[c];
   }
   return status;
}

int NormalizeBase::accumulateSum(pvwdata_t *dataPatchStart, int weights_in_patch, float *sum) {
   // Do not call with sum uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over
   // several patches with multiple calls
   for (int k = 0; k < weights_in_patch; k++) {
      pvwdata_t w = dataPatchStart[k];
      // TODO-CER-2014.4.4 - weight conversion
      *sum += w;
   }
   return PV_SUCCESS;
}

int NormalizeBase::accumulateSumShrunken(
      pvwdata_t *dataPatchStart,
      float *sum,
      int nxpShrunken,
      int nypShrunken,
      int offsetShrunken,
      int xPatchStride,
      int yPatchStride) {
   // Do not call with sumsq uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over
   // several patches with multiple calls
   pvwdata_t *dataPatchStartOffset = dataPatchStart + offsetShrunken;
   int weights_in_row              = xPatchStride * nxpShrunken;
   for (int ky = 0; ky < nypShrunken; ky++) {
      for (int k = 0; k < weights_in_row; k++) {
         pvwdata_t w = dataPatchStartOffset[k];
         *sum += w;
      }
      dataPatchStartOffset += yPatchStride;
   }
   return PV_SUCCESS;
}

int NormalizeBase::accumulateSumSquared(
      pvwdata_t *dataPatchStart,
      int weights_in_patch,
      float *sumsq) {
   // Do not call with sumsq uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over
   // several patches with multiple calls
   for (int k = 0; k < weights_in_patch; k++) {
      pvwdata_t w = dataPatchStart[k];
      *sumsq += w * w;
   }
   return PV_SUCCESS;
}

int NormalizeBase::accumulateSumSquaredShrunken(
      pvwdata_t *dataPatchStart,
      float *sumsq,
      int nxpShrunken,
      int nypShrunken,
      int offsetShrunken,
      int xPatchStride,
      int yPatchStride) {
   // Do not call with sumsq uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over
   // several patches with multiple calls
   pvwdata_t *dataPatchStartOffset = dataPatchStart + offsetShrunken;
   int weights_in_row              = xPatchStride * nxpShrunken;
   for (int ky = 0; ky < nypShrunken; ky++) {
      for (int k = 0; k < weights_in_row; k++) {
         pvwdata_t w = dataPatchStartOffset[k];
         *sumsq += w * w;
      }
      dataPatchStartOffset += yPatchStride;
   }
   return PV_SUCCESS;
}

int NormalizeBase::accumulateMaxAbs(pvwdata_t *dataPatchStart, int weights_in_patch, float *max) {
   // Do not call with max uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over
   // several patches with multiple calls
   float newmax = *max;
   for (int k = 0; k < weights_in_patch; k++) {
      pvwdata_t w = fabsf(dataPatchStart[k]);
      if (w > newmax)
         newmax = w;
   }
   *max = newmax;
   return PV_SUCCESS;
}

int NormalizeBase::accumulateMax(pvwdata_t *dataPatchStart, int weights_in_patch, float *max) {
   // Do not call with max uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over
   // several patches with multiple calls
   float newmax = *max;
   for (int k = 0; k < weights_in_patch; k++) {
      pvwdata_t w = dataPatchStart[k];
      if (w > newmax)
         newmax = w;
   }
   *max = newmax;
   return PV_SUCCESS;
}

int NormalizeBase::accumulateMin(pvwdata_t *dataPatchStart, int weights_in_patch, float *min) {
   // Do not call with min uninitialized.
   // min is cleared inside this routine so that you can accumulate the stats over several patches
   // with multiple calls
   float newmin = *min;
   for (int k = 0; k < weights_in_patch; k++) {
      pvwdata_t w = dataPatchStart[k];
      if (w < newmin)
         newmin = w;
   }
   *min = newmin;
   return PV_SUCCESS;
}

int NormalizeBase::addConnToList(HyPerConn *newConn) {
   HyPerConn **newList = NULL;
   if (connectionList) {
      newList =
            (HyPerConn **)realloc(connectionList, sizeof(*connectionList) * (numConnections + 1));
   }
   else {
      newList = (HyPerConn **)malloc(sizeof(*connectionList) * (numConnections + 1));
   }
   if (newList == NULL) {
      pvError().printf(
            "%s unable to add %s as connection number %d : %s\n",
            getDescription_c(),
            newConn->getDescription_c(),
            numConnections + 1,
            strerror(errno));
   }
   connectionList                 = newList;
   connectionList[numConnections] = newConn;
   numConnections++;
   if (parent->columnId() == 0) {
      pvInfo().printf(
            "Adding %s to normalizer group \"%s\".\n",
            newConn->getDescription_c(),
            this->getName());
   }
   return PV_SUCCESS;
}

void NormalizeBase::normalizePatch(
      pvwdata_t *dataStartPatch,
      int weights_per_patch,
      float multiplier) {
   for (int k = 0; k < weights_per_patch; k++)
      dataStartPatch[k] *= multiplier;
}

} // end namespace PV
