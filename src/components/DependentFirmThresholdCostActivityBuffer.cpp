/*
 * DependentFirmThresholdCostActivityBuffer.cpp
 *
 *  Created on: Apr 2, 2019
 *      Author: pschultz
 */

#include "DependentFirmThresholdCostActivityBuffer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/ActivityComponent.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

DependentFirmThresholdCostActivityBuffer::DependentFirmThresholdCostActivityBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

DependentFirmThresholdCostActivityBuffer::~DependentFirmThresholdCostActivityBuffer() {}

void DependentFirmThresholdCostActivityBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   HyPerActivityBuffer::initialize(name, params, comm);
}

void DependentFirmThresholdCostActivityBuffer::setObjectType() {
   mObjectType = "DependentFirmThresholdCostActivityBuffer";
}

void DependentFirmThresholdCostActivityBuffer::ioParam_VThresh(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(name, "VThresh");
   }
   // During the communication phase, VThresh will be copied from originalConn
}

void DependentFirmThresholdCostActivityBuffer::ioParam_VWidth(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "VWidth", &mVWidth, mVWidth);
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(name, "VWidth");
   }
   // During the communication phase, VWidth will be copied from originalConn
}

Response::Status DependentFirmThresholdCostActivityBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int const maxIterations = 1;
   auto *originalLayerNameParam =
         message->mHierarchy->lookupByTypeRecursive<OriginalLayerNameParam>(maxIterations);
   FatalIf(
         !originalLayerNameParam,
         "%s could not find an OriginalLayerNameParam component.\n",
         getDescription_c());

   if (!originalLayerNameParam->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the OriginalLayerNameParam component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   ComponentBasedObject *originalLayer = nullptr;
   try {
      originalLayer = originalLayerNameParam->findLinkedObject(message->mHierarchy);
   } catch (std::invalid_argument &e) {
      Fatal().printf("%s: %s\n", getDescription_c(), e.what());
   }
   pvAssert(originalLayer);

   auto *originalActivityComponent = originalLayer->getComponentByType<ActivityComponent>();
   FatalIf(
         originalActivityComponent == nullptr,
         "%s original layer \"%s\" does not have an ActivityComponent.\n",
         getDescription_c(),
         originalLayer->getName());

   auto *originalActivityBuffer =
         originalActivityComponent->getComponentByType<ANNActivityBuffer>();
   FatalIf(
         originalActivityBuffer == nullptr,
         "%s original layer \"%s\" does not have an ANNActivityBuffer.\n",
         getDescription_c(),
         originalLayer->getName());
   if (!originalActivityBuffer->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original activity buffer \"%s\" has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c(),
               originalLayer->getName());
      }
      return Response::POSTPONE;
   }

   FatalIf(
         originalActivityBuffer->usingVerticesListInParams(),
         "%s original layer \"%s\" must specify VThresh and VWidth, not verticesV and verticesA.\n",
         getDescription_c(),
         originalLayer->getName());
   mVThresh = originalActivityBuffer->getVThresh();
   mVWidth  = originalActivityBuffer->getVWidth();
   FatalIf(
         originalActivityBuffer->getAMax() < 0.99f * FLT_MAX,
         "%s requires original layer \"%s\" have AMax = infinity; it is %f\n",
         getDescription_c(),
         originalLayer->getName(),
         (double)originalActivityBuffer->getAMax());
   FatalIf(
         originalActivityBuffer->getAMin() != 0.0f,
         "%s requires original layer \"%s\" have AMin = 0; it is %f\n",
         getDescription_c(),
         originalLayer->getName(),
         (double)originalActivityBuffer->getAMin());
   FatalIf(
         originalActivityBuffer->getAShift() != 0.0f,
         "%s requires original layer \"%s\" have AShift = 0; it is %f\n",
         getDescription_c(),
         originalLayer->getName(),
         (double)originalActivityBuffer->getAShift());
   parameters()->handleUnnecessaryParameter(name, "VThresh", mVThresh);
   parameters()->handleUnnecessaryParameter(name, "VWidth", mVWidth);

   auto status = FirmThresholdCostActivityBuffer::communicateInitInfo(message);
   return status;
}

} // namespace PV
