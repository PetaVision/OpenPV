/*
 * BaseDelivery.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "BaseDelivery.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

BaseDelivery::BaseDelivery() {}

BaseDelivery::~BaseDelivery() {}

int BaseDelivery::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

int BaseDelivery::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_channelCode(ioFlag);
   ioParam_delay(ioFlag);

   // GPU-specific parameter.  If not using GPUs, we read it anyway, with
   // warnIfAbsent set to false,
   // to prevent unnecessary warnings from unread or missing parameters.
   ioParam_receiveGpu(ioFlag);

   return PV_SUCCESS;
}

void BaseDelivery::ioParam_channelCode(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      int ch = 0;
      this->parent->parameters()->ioParamValueRequired(ioFlag, this->getName(), "channelCode", &ch);
      int status = decodeChannel(ch, &mChannelCode);
      if (status != PV_SUCCESS) {
         if (this->parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: channelCode %d is not a valid channel.\n", this->getDescription_c(), ch);
         }
         MPI_Barrier(this->parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
   else if (ioFlag == PARAMS_IO_WRITE) {
      int ch = (int)mChannelCode;
      this->parent->parameters()->ioParamValueRequired(ioFlag, this->getName(), "channelCode", &ch);
   }
   else {
      pvAssert(0); // All possibilities of ioFlag are covered above.
   }
}

void BaseDelivery::ioParam_delay(enum ParamsIOFlag ioFlag) {
   // Grab delays in ms and load into mDelayFromParams.
   // initializeDelays() will convert the delays to timesteps store into delays.
   double *delayArray;
   int delayArraySize;
   if (ioFlag == PARAMS_IO_WRITE) {
      delayArray     = mDelayFromParams.data();
      delayArraySize = (int)mDelayFromParams.size();
   }
   this->parent->parameters()->ioParamArray(
         ioFlag, this->getName(), "delay", &delayArray, &delayArraySize);
   if (ioFlag == PARAMS_IO_READ) {
      if (delayArraySize == 0) {
         mDelayFromParams.resize(1);
         mDelayFromParams[0] = 0.0f;
         if (this->parent->columnId() == 0) {
            InfoLog() << getDescription() << ": Using default value of zero for delay.\n";
         }
      }
      else {
         mDelayFromParams.resize(delayArraySize);
         for (int d = 0; d < delayArraySize; d++) {
            mDelayFromParams[d] = delayArray[d];
         }
      }
   }
}

void BaseDelivery::ioParam_receiveGpu(enum ParamsIOFlag ioFlag) {
#ifdef PV_USE_CUDA
   parent->parameters()->ioParamValue(
         ioFlag, name, "receiveGpu", &mReceiveGpu, false /*default*/, true /*warn if absent*/);
   mUsingGPUFlag = mReceiveGpu;
#else
   mReceiveGpu = false;
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

void BaseDelivery::setNumArbors(int numArbors) {
   FatalIf(
         mNumArbors > 0,
         "Setting number of arbors in %s, but it was already set (to %d).\n",
         getDescription_c(),
         mNumArbors);
   FatalIf(
         numArbors <= 0,
         "%s setNumArbors called with value %d. The number of arbors must be positive.\n",
         getDescription_c(),
         numArbors);
   mNumArbors = numArbors;
}

int BaseDelivery::allocateDataStructures() {
   int status = BaseObject::allocateDataStructures();
   FatalIf(
         mNumArbors <= 0,
         "%s delivery component's allocateDataStructures was called before setNumArbors.\n");
   initializeDelays();
   return status;
}

void BaseDelivery::setPreAndPostLayers(HyPerLayer *preLayer, HyPerLayer *postLayer) {
   FatalIf(
         preLayer == nullptr,
         "%s delivery component set the presynaptic layer to the null pointer.\n",
         getDescription_c());
   FatalIf(
         postLayer == nullptr,
         "%s delivery component set the postsynaptic layer to the null pointer.\n",
         getDescription_c());
   FatalIf(
         preLayer->getLayerLoc()->nbatch != postLayer->getLayerLoc()->nbatch,
         "%s called with pre and post layers with different batch size (%d versus %d)\n",
         getDescription_c(),
         preLayer->getLayerLoc()->nbatch,
         postLayer->getLayerLoc()->nbatch);
   mPreLayer  = preLayer;
   mPostLayer = postLayer;
}

void BaseDelivery::initializeDelays() {
   pvAssert(mNumArbors > 0);
   mDelay.resize(mNumArbors);
   double deltaTime                     = parent->getDeltaTime();
   std::size_t const numDelayFromParams = mDelayFromParams.size();
   if (numDelayFromParams == (std::size_t)1) {
      int const delayFromParams = convertToNumberOfTimesteps(mDelayFromParams[0], deltaTime);
      for (auto &d : mDelay) {
         d = delayFromParams;
      }
   }
   else if (numDelayFromParams == mDelay.size()) {
      for (std::size_t k = 0; k < numDelayFromParams; k++) {
         mDelay[k] = convertToNumberOfTimesteps(mDelayFromParams[k], deltaTime);
      }
   }
   else {
      Fatal().printf(
            "%s: delay must be either a single value or the same length as the number of arbors\n",
            getDescription_c());
   }
}

int BaseDelivery::convertToNumberOfTimesteps(double delay, double deltaTime) {
   int intDelay = (int)std::round(delay / deltaTime);
   if (fmod(delay, deltaTime) != 0) {
      double roundedDelay = intDelay * parent->getDeltaTime();
      WarnLog() << getName() << ": A delay of " << delay << " will be rounded to " << roundedDelay
                << "\n";
   }
   return intDelay;
}

void BaseDelivery::deliver(Weights *weights) {}

} // end namespace PV
