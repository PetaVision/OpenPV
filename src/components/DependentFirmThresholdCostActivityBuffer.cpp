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
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

DependentFirmThresholdCostActivityBuffer::~DependentFirmThresholdCostActivityBuffer() {}

void DependentFirmThresholdCostActivityBuffer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerActivityBuffer::initialize(params, defaults, comm);
}

void DependentFirmThresholdCostActivityBuffer::setObjectType() {
   mObjectType = "DependentFirmThresholdCostActivityBuffer";
}

void DependentFirmThresholdCostActivityBuffer::ioParam_VThresh(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      mParamsIO->handleUnnecessaryParameter("VThresh");
   }
   // During the communication phase, VThresh will be copied from originalConn
}

void DependentFirmThresholdCostActivityBuffer::ioParam_VWidth(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      mParamsIO->handleUnnecessaryParameter("VWidth");
   }
   // During the communication phase, VWidth will be copied from originalConn
}

Response::Status DependentFirmThresholdCostActivityBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *objectTable = message->mObjectTable;

   auto *originalLayerNameParam = objectTable->findObject<OriginalLayerNameParam>(getName());
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

   std::string const &linkedObjectName = originalLayerNameParam->getLinkedObjectName();
   auto *originalActivityBuffer = objectTable->findObject<ANNActivityBuffer>(linkedObjectName);
   FatalIf(
         originalActivityBuffer == nullptr,
         "%s original layer \"%s\" does not have an ANNActivityBuffer.\n",
         getDescription_c(),
         linkedObjectName.c_str());
   if (!originalActivityBuffer->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original activity buffer \"%s\" has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c(),
               linkedObjectName.c_str());
      }
      return Response::POSTPONE;
   }

   FatalIf(
         originalActivityBuffer->usingVerticesListInParams(),
         "%s original layer \"%s\" must specify VThresh and VWidth, not verticesV and verticesA.\n",
         getDescription_c(),
         linkedObjectName);
   mVThresh = originalActivityBuffer->getVThresh();
   mVWidth  = originalActivityBuffer->getVWidth();
   FatalIf(
         originalActivityBuffer->getAMax() < 0.99f * FLT_MAX,
         "%s requires original layer \"%s\" have AMax = infinity; it is %f\n",
         getDescription_c(),
         linkedObjectName,
         (double)originalActivityBuffer->getAMax());
   FatalIf(
         originalActivityBuffer->getAMin() != 0.0f,
         "%s requires original layer \"%s\" have AMin = 0; it is %f\n",
         getDescription_c(),
         linkedObjectName,
         (double)originalActivityBuffer->getAMin());
   FatalIf(
         originalActivityBuffer->getAShift() != 0.0f,
         "%s requires original layer \"%s\" have AShift = 0; it is %f\n",
         getDescription_c(),
         linkedObjectName,
         (double)originalActivityBuffer->getAShift());
   mParamsIO->handleUnnecessaryParameter("VThresh", mVThresh);
   mParamsIO->handleUnnecessaryParameter("VWidth", mVWidth);

   auto status = FirmThresholdCostActivityBuffer::communicateInitInfo(message);
   return status;
}

} // namespace PV
