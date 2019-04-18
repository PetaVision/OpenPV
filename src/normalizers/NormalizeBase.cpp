/*
 * NormalizeBase.cpp
 *
 *  Created on: Apr 5, 2013
 *      Author: Pete Schultz
 */

#include "NormalizeBase.hpp"
#include "columns/HyPerCol.hpp"
#include "components/StrengthParam.hpp"
#include "components/WeightsPair.hpp"
#include "layers/HyPerLayer.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

NormalizeBase::NormalizeBase(char const *name, HyPerCol *hc) { initialize(name, hc); }

int NormalizeBase::initialize(char const *name, HyPerCol *hc) {
   int status = BaseObject::initialize(name, hc);
   return status;
}

void NormalizeBase::setObjectType() {
   auto *params                = parent->parameters();
   char const *normalizeMethod = params->stringValue(name, "normalizeMethod", false);
   mObjectType                 = normalizeMethod ? normalizeMethod : "Normalizer for";
}

int NormalizeBase::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_normalizeMethod(ioFlag);
   ioParam_normalizeArborsIndividually(ioFlag);
   ioParam_normalizeOnInitialize(ioFlag);
   ioParam_normalizeOnWeightUpdate(ioFlag);
   return PV_SUCCESS;
}

void NormalizeBase::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(ioFlag, name, "normalizeMethod", &mNormalizeMethod);
}

void NormalizeBase::ioParam_normalizeArborsIndividually(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "normalizeArborsIndividually",
         &mNormalizeArborsIndividually,
         mNormalizeArborsIndividually,
         true /*warnIfAbsent*/);
}

void NormalizeBase::ioParam_normalizeOnInitialize(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "normalizeOnInitialize", &mNormalizeOnInitialize, mNormalizeOnInitialize);
}

void NormalizeBase::ioParam_normalizeOnWeightUpdate(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "normalizeOnWeightUpdate",
         &mNormalizeOnWeightUpdate,
         mNormalizeOnWeightUpdate);
}

Response::Status NormalizeBase::respond(std::shared_ptr<BaseMessage const> message) {
   Response::Status status = BaseObject::respond(message);
   if (status != Response::SUCCESS) {
      return status;
   }
   else if (
         auto castMessage = std::dynamic_pointer_cast<ConnectionNormalizeMessage const>(message)) {
      return respondConnectionNormalize(castMessage);
   }
   else {
      return status;
   }
}

Response::Status NormalizeBase::respondConnectionNormalize(
      std::shared_ptr<ConnectionNormalizeMessage const> message) {
   bool needUpdate = false;
   double simTime  = parent->simulationTime();
   if (mNormalizeOnInitialize && simTime == 0.0) {
      needUpdate = true;
   }
   else if (mNormalizeOnWeightUpdate and weightsHaveUpdated()) {
      needUpdate = true;
   }
   if (needUpdate) {
      normalizeWeights();
      mLastTimeNormalized = simTime;
      for (auto &w : mWeightsList) {
         pvAssert(w);
         w->setTimestamp(simTime);
      }
      return Response::SUCCESS;
   }
   else {
      return Response::NO_ACTION;
   }
}

Response::Status
NormalizeBase::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *weightsPair = mapLookupByType<WeightsPair>(message->mHierarchy, getDescription());
   pvAssert(weightsPair);
   if (!weightsPair->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   auto *strengthParam = mapLookupByType<StrengthParam>(message->mHierarchy, getDescription());
   pvAssert(strengthParam);
   if (!strengthParam->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   mStrength = strengthParam->getStrength();

   auto status = BaseObject::communicateInitInfo(message);
   if (status != Response::SUCCESS) {
      return status;
   }

   weightsPair->needPre();
   Weights *weights = weightsPair->getPreWeights();
   pvAssert(weights != nullptr);
   addWeightsToList(weights);

   return Response::SUCCESS;
}

void NormalizeBase::addWeightsToList(Weights *weights) {
   mWeightsList.push_back(weights);
   if (parent->getCommunicator()->globalCommRank() == 0) {
      InfoLog().printf(
            "Adding %s to normalizer group \"%s\".\n", weights->getName().c_str(), this->getName());
   }
}

bool NormalizeBase::weightsHaveUpdated() const {
   bool haveUpdated = false;
   for (auto &w : mWeightsList) {
      pvAssert(w);
      if (w->getTimestamp() > mLastTimeNormalized) {
         haveUpdated = true;
         break;
      }
   }
   return haveUpdated;
}

int NormalizeBase::accumulateSum(float *dataPatchStart, int weights_in_patch, float *sum) {
   // Do not call with sum uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over
   // several patches with multiple calls
   for (int k = 0; k < weights_in_patch; k++) {
      float w = dataPatchStart[k];
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

} // namespace PV
