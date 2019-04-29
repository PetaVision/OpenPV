/*
 * ArborList.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#include "ArborList.hpp"
#include "components/ConnectionData.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

ArborList::ArborList(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

ArborList::ArborList() {}

ArborList::~ArborList() { free(mDelaysParams); }

void ArborList::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

void ArborList::setObjectType() { mObjectType = "ArborList"; }

int ArborList::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_numAxonalArbors(ioFlag);
   ioParam_delay(ioFlag);
   return PV_SUCCESS;
}

void ArborList::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, this->getName(), "numAxonalArbors", &mNumAxonalArbors, mNumAxonalArbors);
   if (ioFlag == PARAMS_IO_READ) {
      if (getNumAxonalArbors() <= 0 && mCommunicator->globalCommRank() == 0) {
         WarnLog().printf(
               "Connection %s: Variable numAxonalArbors is set to 0. "
               "No connections will be made.\n",
               this->getName());
      }
   }
}

void ArborList::ioParam_delay(enum ParamsIOFlag ioFlag) {
   // Grab delays in ms and load into mDelaysParams.
   // initializeDelays() will convert the delays to timesteps store into delays.
   parameters()->ioParamArray(ioFlag, getName(), "delay", &mDelaysParams, &mNumDelays);
   if (ioFlag == PARAMS_IO_READ && mNumDelays == 0) {
      assert(mDelaysParams == nullptr);
      mDelaysParams = (double *)pvMallocError(
            sizeof(double),
            "%s: unable to set default delay: %s\n",
            this->getDescription_c(),
            strerror(errno));
      *mDelaysParams = 0.0f; // Default delay
      mNumDelays     = 1;
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf("%s: Using default value of zero for delay.\n", this->getDescription_c());
      }
   }
}

Response::Status
ArborList::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *connectionData = message->mObjectTable->findObject<ConnectionData>(getName());
   pvAssert(connectionData);

   if (!connectionData->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the ConnectionData component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }
   HyPerLayer *preLayer            = connectionData->getPre();
   BasePublisherComponent *preData = preLayer->getComponentByType<BasePublisherComponent>();

   initializeDelays(message->mDeltaTime);
   int maxDelay     = maxDelaySteps();
   int allowedDelay = preData->increaseDelayLevels(maxDelay);
   if (allowedDelay < maxDelay) {
      if (mCommunicator->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: attempt to set delay to %d, but the maximum "
               "allowed delay is %d.  Exiting\n",
               getDescription_c(),
               maxDelay,
               allowedDelay);
      }
      MPI_Barrier(mCommunicator->globalCommunicator());
      exit(EXIT_FAILURE);
   }

   return Response::SUCCESS;
}

void ArborList::initializeDelays(double deltaTime) {
   assert(!parameters()->presentAndNotBeenRead(this->getName(), "numAxonalArbors"));
   mDelay.resize(getNumAxonalArbors());

   FatalIf(
         mNumDelays != 1 and mNumDelays != getNumAxonalArbors(),
         "%s Delay must be either a single value or the same length as the number of arbors\n",
         getDescription_c());
   for (int arborId = 0; arborId < (int)mDelay.size(); arborId++) {
      int delayIndex  = (mNumDelays == getNumAxonalArbors()) ? arborId : 0;
      mDelay[arborId] = convertDelay(mDelaysParams[delayIndex], deltaTime);
   }
}

int ArborList::convertDelay(double delay, double deltaTime) {
   int intDelay = (int)std::nearbyint(delay / deltaTime);
   if (std::fmod(delay, deltaTime) != 0) {
      double actualDelay = intDelay * deltaTime;
      WarnLog() << getName() << ": A delay of " << delay << " will be rounded to " << actualDelay
                << "\n";
   }
   return intDelay;
}

int ArborList::maxDelaySteps() {
   int maxDelay = 0;
   for (auto &d : mDelay) {
      if (d > maxDelay) {
         maxDelay = d;
      }
   }
   return maxDelay;
}

} // namespace PV
