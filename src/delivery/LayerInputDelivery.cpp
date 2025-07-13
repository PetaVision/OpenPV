/*
 * LayerInputDelivery.cpp
 *
 *  Created on: Sept 17, 2018
 *      Author: Pete Schultz
 */

#include "LayerInputDelivery.hpp"

namespace PV {

LayerInputDelivery::LayerInputDelivery(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

void LayerInputDelivery::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   mMPIReductionOp = MPI_SUM; // This can be changed by derived classes if needed.
   BaseObject::initialize(params, defaults, comm);
}

void LayerInputDelivery::setObjectType() { mObjectType = "LayerInputDelivery"; }

int LayerInputDelivery::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_channelCode(ioSwitch);
   ioParam_receiveGpu(ioSwitch);
   return PV_SUCCESS;
}

void LayerInputDelivery::ioParam_channelCode(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      int ch = 0;
      mParamsIO->ioParam(ioSwitch, "channelCode", &ch);
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
   else if (ioSwitch == ParamsIOSwitch::Write) {
      int ch = (int)mChannelCode;
      mParamsIO->ioParam(ioSwitch, "channelCode", &ch);
   }
   else {
      assert(0); // All possibilities of ioSwitch are covered above.
   }
}

void LayerInputDelivery::ioParam_receiveGpu(ParamsIOSwitch ioSwitch) {
#ifdef PV_USE_CUDA
   mParamsIO->ioParam(ioSwitch, "receiveGpu", &mReceiveGpu);
#else
   mParamsIO->ioParam(ioSwitch, "receiveGpu", &mReceiveGpu, false /*warnIfAbsentFlag*/);
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
