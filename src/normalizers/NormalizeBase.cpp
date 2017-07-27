/*
 * NormalizeBase.cpp
 *
 *  Created on: Apr 5, 2013
 *      Author: pschultz
 */

#include "NormalizeBase.hpp"

namespace PV {

NormalizeBase::NormalizeBase() { initialize_base(); }

NormalizeBase::~NormalizeBase() {}

int NormalizeBase::initialize_base() {
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

   int status = BaseObject::initialize(name, hc);
   status     = parent->addNormalizer(this);
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

int NormalizeBase::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   HyPerConn *conn = message->lookup<HyPerConn>(std::string(name));
   pvAssertMessage(conn != nullptr, "No connection \"%s\" for %s.\n", name, getDescription_c());
   pvAssert(conn != nullptr);
   return addConnToList(conn);
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
      for (auto &c : connectionList) {
         if (simTime == c->getLastUpdateTime()) {
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
   for (auto &c : connectionList) {
   }
   return status;
}

int NormalizeBase::accumulateSum(float *dataPatchStart, int weights_in_patch, float *sum) {
   // Do not call with sum uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over
   // several patches with multiple calls
   for (int k = 0; k < weights_in_patch; k++) {
      float w = dataPatchStart[k];
      // TODO-CER-2014.4.4 - weight conversion
      *sum += w;
   }
   return PV_SUCCESS;
}

int NormalizeBase::accumulateSumShrunken(
      float *dataPatchStart,
      float *sum,
      int nxpShrunken,
      int nypShrunken,
      int offsetShrunken,
      int xPatchStride,
      int yPatchStride) {
   // Do not call with sumsq uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over
   // several patches with multiple calls
   float *dataPatchStartOffset = dataPatchStart + offsetShrunken;
   int weights_in_row          = xPatchStride * nxpShrunken;
   for (int ky = 0; ky < nypShrunken; ky++) {
      for (int k = 0; k < weights_in_row; k++) {
         float w = dataPatchStartOffset[k];
         *sum += w;
      }
      dataPatchStartOffset += yPatchStride;
   }
   return PV_SUCCESS;
}

int NormalizeBase::accumulateSumSquared(float *dataPatchStart, int weights_in_patch, float *sumsq) {
   // Do not call with sumsq uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over
   // several patches with multiple calls
   for (int k = 0; k < weights_in_patch; k++) {
      float w = dataPatchStart[k];
      *sumsq += w * w;
   }
   return PV_SUCCESS;
}

int NormalizeBase::accumulateSumSquaredShrunken(
      float *dataPatchStart,
      float *sumsq,
      int nxpShrunken,
      int nypShrunken,
      int offsetShrunken,
      int xPatchStride,
      int yPatchStride) {
   // Do not call with sumsq uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over
   // several patches with multiple calls
   float *dataPatchStartOffset = dataPatchStart + offsetShrunken;
   int weights_in_row          = xPatchStride * nxpShrunken;
   for (int ky = 0; ky < nypShrunken; ky++) {
      for (int k = 0; k < weights_in_row; k++) {
         float w = dataPatchStartOffset[k];
         *sumsq += w * w;
      }
      dataPatchStartOffset += yPatchStride;
   }
   return PV_SUCCESS;
}

int NormalizeBase::accumulateMaxAbs(float *dataPatchStart, int weights_in_patch, float *max) {
   // Do not call with max uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over
   // several patches with multiple calls
   float newmax = *max;
   for (int k = 0; k < weights_in_patch; k++) {
      float w = fabsf(dataPatchStart[k]);
      if (w > newmax)
         newmax = w;
   }
   *max = newmax;
   return PV_SUCCESS;
}

int NormalizeBase::accumulateMax(float *dataPatchStart, int weights_in_patch, float *max) {
   // Do not call with max uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over
   // several patches with multiple calls
   float newmax = *max;
   for (int k = 0; k < weights_in_patch; k++) {
      float w = dataPatchStart[k];
      if (w > newmax)
         newmax = w;
   }
   *max = newmax;
   return PV_SUCCESS;
}

int NormalizeBase::accumulateMin(float *dataPatchStart, int weights_in_patch, float *min) {
   // Do not call with min uninitialized.
   // min is cleared inside this routine so that you can accumulate the stats over several patches
   // with multiple calls
   float newmin = *min;
   for (int k = 0; k < weights_in_patch; k++) {
      float w = dataPatchStart[k];
      if (w < newmin)
         newmin = w;
   }
   *min = newmin;
   return PV_SUCCESS;
}

int NormalizeBase::addConnToList(HyPerConn *newConn) {
   connectionList.push_back(newConn);
   if (parent->columnId() == 0) {
      InfoLog().printf(
            "Adding %s to normalizer group \"%s\".\n",
            newConn->getDescription_c(),
            this->getName());
   }
   return PV_SUCCESS;
}

void NormalizeBase::normalizePatch(float *dataStartPatch, int weights_per_patch, float multiplier) {
   for (int k = 0; k < weights_per_patch; k++)
      dataStartPatch[k] *= multiplier;
}

} // end namespace PV
