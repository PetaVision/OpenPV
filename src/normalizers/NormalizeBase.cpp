/*
 * NormalizeBase.cpp
 *
 *  Created on: Apr 5, 2013
 *      Author: Pete Schultz
 */

#include "NormalizeBase.hpp"
#include "components/WeightsPair.hpp"
#include "layers/HyPerLayer.hpp"
#include "structures/Weights.hpp"
#include "utils/PVAssert.hpp"

namespace PV {

NormalizeBase::NormalizeBase(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

NormalizeBase::~NormalizeBase() {}

void NormalizeBase::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   BaseObject::initialize(paramsIO, comm);
}

void NormalizeBase::initMessageActionMap() {
   BaseObject::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ConnectionNormalizeMessage const>(msgptr);
      return respondConnectionNormalize(castMessage);
   };
   mMessageActionMap.emplace("ConnectionNormalize", action);
}

void NormalizeBase::setObjectType() {
   std::string const *objectTypePtr = mParamsIO->getParams()->read<std::string>("normalizeMethod");
   FatalIf(
         objectTypePtr == nullptr or objectTypePtr->empty(),
         "normalizeMethod for parameter group \"%s\" cannot be NULL or empty.\n", getName());
   mObjectType = *objectTypePtr;
}

int NormalizeBase::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_normalizeMethod(ioSwitch);
   ioParam_normalizeArborsIndividually(ioSwitch);
   ioParam_normalizeOnInitialize(ioSwitch);
   ioParam_normalizeOnWeightUpdate(ioSwitch);
   return PV_SUCCESS;
}

void NormalizeBase::ioParam_normalizeMethod(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "normalizeMethod", &mNormalizeMethod);
}

void NormalizeBase::ioParam_normalizeArborsIndividually(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "normalizeArborsIndividually", &mNormalizeArborsIndividually);
}

void NormalizeBase::ioParam_normalizeOnInitialize(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "normalizeOnInitialize", &mNormalizeOnInitialize);
}

void NormalizeBase::ioParam_normalizeOnWeightUpdate(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("normalizeOnInitialize"));
   mParamsIO->ioParam(ioSwitch, "normalizeOnWeightUpdate", &mNormalizeOnWeightUpdate);
   FatalIf(
         mNormalizeOnInitialize == false and mNormalizeOnWeightUpdate == true,
         "Connection \"%s\": normalizeOnInitialize cannot be false if "
         "normalizeOnWeightUpdate is true.\n",
         getName());
   if (mNormalizeOnInitialize == false and mNormalizeOnWeightUpdate == false) {
      WarnLog().printf(
            "Connection \"%s\" has both normalizationOnInitialize and normalizeOnWeightUpdate"
            "set to false. No normalization will take place.\n",
            getName());
   }
}

Response::Status NormalizeBase::respondConnectionNormalize(
      std::shared_ptr<ConnectionNormalizeMessage const> message) {
   bool needUpdate = false;
   double simTime  = message->mTime;
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

StrengthParam *NormalizeBase::retrieveStrengthParamIfNeeded(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return nullptr;
}

Response::Status
NormalizeBase::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *weightsPair = message->mObjectTable->findObject<WeightsPair>(getName());
   pvAssert(weightsPair);
   if (!weightsPair->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   mConnectionData = message->mObjectTable->findObject<ConnectionData>(getName());
   pvAssert(mConnectionData);
   if (!mConnectionData->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   auto *strengthParam = retrieveStrengthParamIfNeeded(message);
   // retrieveStrengthParamIfNeeded() is a virtual function member that returns nullptr
   // for derived classes that do not use the strength parameter, and a valid pointer
   // for derived classes that do.
   if (strengthParam) {
      if (!strengthParam->getInitInfoCommunicatedFlag()) {
         return Response::POSTPONE;
      }
      mStrength = strengthParam->getStrength();
   }

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
   if (mCommunicator->globalCommRank() == 0) {
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
