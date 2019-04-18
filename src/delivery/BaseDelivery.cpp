/*
 * BaseDelivery.cpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#include "BaseDelivery.hpp"
#include "columns/HyPerCol.hpp"
#include "layers/HyPerLayer.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

BaseDelivery::BaseDelivery(char const *name, HyPerCol *hc) { initialize(name, hc); }

int BaseDelivery::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

void BaseDelivery::setObjectType() { mObjectType = "BaseDelivery"; }

int BaseDelivery::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_channelCode(ioFlag);
   ioParam_receiveGpu(ioFlag);
   return PV_SUCCESS;
}

void BaseDelivery::ioParam_channelCode(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      int ch = 0;
      this->parent->parameters()->ioParamValueRequired(ioFlag, this->getName(), "channelCode", &ch);
      switch (ch) {
         case CHANNEL_EXC: mChannelCode      = CHANNEL_EXC; break;
         case CHANNEL_INH: mChannelCode      = CHANNEL_INH; break;
         case CHANNEL_INHB: mChannelCode     = CHANNEL_INHB; break;
         case CHANNEL_GAP: mChannelCode      = CHANNEL_GAP; break;
         case CHANNEL_NORM: mChannelCode     = CHANNEL_NORM; break;
         case CHANNEL_NOUPDATE: mChannelCode = CHANNEL_NOUPDATE; break;
         default:
            if (parent->getCommunicator()->globalCommRank() == 0) {
               ErrorLog().printf(
                     "%s: channelCode %d is not a valid channel.\n", this->getDescription_c(), ch);
            }
            MPI_Barrier(this->parent->getCommunicator()->globalCommunicator());
            exit(EXIT_FAILURE);
            break;
      }
   }
   else if (ioFlag == PARAMS_IO_WRITE) {
      int ch = (int)mChannelCode;
      parent->parameters()->ioParamValueRequired(ioFlag, this->getName(), "channelCode", &ch);
   }
   else {
      assert(0); // All possibilities of ioFlag are covered above.
   }
}

void BaseDelivery::ioParam_receiveGpu(enum ParamsIOFlag ioFlag) {
#ifdef PV_USE_CUDA
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "receiveGpu",
         &mReceiveGpu,
         mReceiveGpu /*default*/,
         true /*warn if absent*/);
#else
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "receiveGpu",
         &mReceiveGpu,
         mReceiveGpu /*default*/,
         false /*warn if absent*/);
   if (parent->getCommunicator()->globalCommRank() == 0) {
      FatalIf(
            mReceiveGpu,
            "%s: receiveGpu is set to true in params, but PetaVision was compiled without GPU "
            "acceleration.\n",
            getDescription_c());
   }
#endif // PV_USE_CUDA
}

Response::Status
BaseDelivery::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (mConnectionData == nullptr) {
      mConnectionData = mapLookupByType<ConnectionData>(message->mHierarchy, getDescription());
      FatalIf(
            mConnectionData == nullptr,
            "%s requires a ConnectionData component.\n",
            getDescription_c());
   }
   pvAssert(mConnectionData);
   if (!mConnectionData->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the ConnectionData component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   mPreLayer  = mConnectionData->getPre();
   mPostLayer = mConnectionData->getPost();
   pvAssert(mPreLayer != nullptr and mPostLayer != nullptr);

   int numChannelsCheck = 0;
   int channelAsInt     = (int)getChannelCode();
   if (channelAsInt >= 0) {
      int status = getPostLayer()->requireChannel(channelAsInt, &numChannelsCheck);
      if (status != PV_SUCCESS) {
         if (parent->getCommunicator()->globalCommRank() == 0) {
            ErrorLog().printf(
                  "%s: postsynaptic layer \"%s\" failed to add channel %d\n",
                  getDescription_c(),
                  getPostLayer()->getName(),
                  channelAsInt);
         }
         MPI_Barrier(parent->getCommunicator()->globalCommunicator());
         exit(EXIT_FAILURE);
      }
   }
#ifdef PV_USE_CUDA
   mUsingGPUFlag = mReceiveGpu;
#endif // PV_USE_CUDA

   return Response::SUCCESS;
}

} // namespace PV
