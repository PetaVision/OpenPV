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

int BaseDelivery::setDescription() {
   description.clear();
   description.append("BaseDelivery").append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

int BaseDelivery::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_channelCode(ioFlag);
   ioParam_delay(ioFlag);
   ioParam_convertRateToSpikeCount(ioFlag);
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

void BaseDelivery::ioParam_delay(enum ParamsIOFlag ioFlag) {
   // Grab delays in ms and load into mDelaysParams.
   // initializeDelays() will convert the delays to timesteps store into delays.
   parent->parameters()->ioParamArray(ioFlag, getName(), "delay", &mDelaysParams, &mNumDelays);
   if (ioFlag == PARAMS_IO_READ && mNumDelays == 0) {
      assert(mDelaysParams == nullptr);
      mDelaysParams = (double *)pvMallocError(
            sizeof(double),
            "%s: unable to set default delay: %s\n",
            this->getDescription_c(),
            strerror(errno));
      *mDelaysParams = 0.0f; // Default delay
      mNumDelays     = 1;
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf("%s: Using default value of zero for delay.\n", this->getDescription_c());
      }
   }
}

void BaseDelivery::ioParam_convertRateToSpikeCount(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         getName(),
         "convertRateToSpikeCount",
         &mConvertRateToSpikeCount,
         false /*default value*/);
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
   if (parent->columnId() == 0) {
      FatalIf(
            mReceiveGpu,
            "%s: receiveGpu is set to true in params, but PetaVision was compiled without GPU "
            "acceleration.\n",
            getDescription_c());
   }
#endif // PV_USE_CUDA
}

int BaseDelivery::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   pvAssert(mConnectionData == nullptr);
   mConnectionData = mapLookupByType<ConnectionData>(message->mHierarchy, getDescription());
   pvAssert(mConnectionData != nullptr);

   if (!mConnectionData->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the ConnectionData component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return PV_POSTPONE;
   }

   mPreLayer  = mConnectionData->getPre();
   mPostLayer = mConnectionData->getPost();

   initializeDelays();
   int maxDelay     = maxDelaySteps();
   int allowedDelay = getPreLayer()->increaseDelayLevels(maxDelay);
   if (allowedDelay < maxDelay) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: attempt to set delay to %d, but the maximum "
               "allowed delay is %d.  Exiting\n",
               getDescription_c(),
               maxDelay,
               allowedDelay);
      }
      exit(EXIT_FAILURE);
   }

   return PV_SUCCESS;
}

void BaseDelivery::initializeDelays() {
   assert(!parent->parameters()->presentAndNotBeenRead(this->getName(), "numAxonalArbors"));
   mDelay.resize(mConnectionData->getNumAxonalArbors());

   // Initialize delays for each arbor
   // Using setDelay to convert ms to timesteps
   for (int arborId = 0; arborId < (int)mDelay.size(); arborId++) {
      if (mNumDelays == 0) {
         // No delay
         setDelay(arborId, 0.0);
      }
      else if (mNumDelays == 1) {
         setDelay(arborId, mDelaysParams[0]);
      }
      else if (mNumDelays == mConnectionData->getNumAxonalArbors()) {
         setDelay(arborId, mDelaysParams[arborId]);
      }
      else {
         Fatal().printf(
               "Delay must be either a single value or the same length "
               "as the number of arbors\n");
      }
   }
}

void BaseDelivery::setDelay(int arborId, double delay) {
   assert(arborId >= 0 && arborId < mConnectionData->getNumAxonalArbors());
   int intDelay = (int)std::nearbyint(delay / parent->getDeltaTime());
   if (std::fmod(delay, parent->getDeltaTime()) != 0) {
      double actualDelay = intDelay * parent->getDeltaTime();
      WarnLog() << getName() << ": A delay of " << delay << " will be rounded to " << actualDelay
                << "\n";
   }
   mDelay[arborId] = intDelay;
}

int BaseDelivery::maxDelaySteps() {
   int maxDelay        = 0;
   int const numArbors = mConnectionData->getNumAxonalArbors();
   for (auto &d : mDelay) {
      if (d > maxDelay) {
         maxDelay = d;
      }
   }
   return maxDelay;
}

} // namespace PV