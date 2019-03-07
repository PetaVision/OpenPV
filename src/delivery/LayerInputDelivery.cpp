/*
 * LayerInputDelivery.cpp
 *
 *  Created on: Sept 17, 2018
 *      Author: Pete Schultz
 */

#include "LayerInputDelivery.hpp"

namespace PV {

LayerInputDelivery::LayerInputDelivery(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

void LayerInputDelivery::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

void LayerInputDelivery::setObjectType() { mObjectType = "LayerInputDelivery"; }

int LayerInputDelivery::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_channelCode(ioFlag);
   ioParam_receiveGpu(ioFlag);
   return PV_SUCCESS;
}

void LayerInputDelivery::ioParam_channelCode(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      int ch = 0;
      this->parameters()->ioParamValueRequired(ioFlag, this->getName(), "channelCode", &ch);
      switch (ch) {
         case CHANNEL_EXC: mChannelCode      = CHANNEL_EXC; break;
         case CHANNEL_INH: mChannelCode      = CHANNEL_INH; break;
         case CHANNEL_INHB: mChannelCode     = CHANNEL_INHB; break;
         case CHANNEL_GAP: mChannelCode      = CHANNEL_GAP; break;
         case CHANNEL_NORM: mChannelCode     = CHANNEL_NORM; break;
         case CHANNEL_NOUPDATE: mChannelCode = CHANNEL_NOUPDATE; break;
         default:
            if (mCommunicator->globalCommRank() == 0) {
               ErrorLog().printf(
                     "%s: channelCode %d is not a valid channel.\n", this->getDescription_c(), ch);
            }
            MPI_Barrier(this->mCommunicator->globalCommunicator());
            exit(EXIT_FAILURE);
            break;
      }
   }
   else if (ioFlag == PARAMS_IO_WRITE) {
      int ch = (int)mChannelCode;
      parameters()->ioParamValueRequired(ioFlag, this->getName(), "channelCode", &ch);
   }
   else {
      assert(0); // All possibilities of ioFlag are covered above.
   }
}

void LayerInputDelivery::ioParam_receiveGpu(enum ParamsIOFlag ioFlag) {
#ifdef PV_USE_CUDA
   parameters()->ioParamValue(
         ioFlag,
         name,
         "receiveGpu",
         &mReceiveGpu,
         mReceiveGpu /*default*/,
         true /*warn if absent*/);
#else
   parameters()->ioParamValue(
         ioFlag,
         name,
         "receiveGpu",
         &mReceiveGpu,
         mReceiveGpu /*default*/,
         false /*warn if absent*/);
   if (mCommunicator->globalCommRank() == 0) {
      FatalIf(
            mReceiveGpu,
            "%s: receiveGpu is set to true in params, but PetaVision was compiled without GPU "
            "acceleration.\n",
            getDescription_c());
   }
#endif // PV_USE_CUDA
}

} // namespace PV
