/*
 * ArborList.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#include "ArborList.hpp"
#include "columns/HyPerCol.hpp"
#include "components/ConnectionData.hpp"
#include "layers/HyPerLayer.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

ArborList::ArborList(char const *name, HyPerCol *hc) { initialize(name, hc); }

ArborList::ArborList() {}

ArborList::~ArborList() { free(mDelaysParams); }

int ArborList::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

void ArborList::setObjectType() { mObjectType = "ArborList"; }

int ArborList::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_numAxonalArbors(ioFlag);
   ioParam_delay(ioFlag);
   return PV_SUCCESS;
}

void ArborList::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, this->getName(), "numAxonalArbors", &mNumAxonalArbors, mNumAxonalArbors);
   if (ioFlag == PARAMS_IO_READ) {
      if (getNumAxonalArbors() <= 0 && parent->getCommunicator()->globalCommRank() == 0) {
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

Response::Status
ArborList::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   ConnectionData *connectionData =
         mapLookupByType<ConnectionData>(message->mHierarchy, getDescription());
   FatalIf(
         connectionData == nullptr,
         "%s received CommunicateInitInfo message without a ConnectionData component.\n",
         getDescription_c());

   if (!connectionData->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the ConnectionData component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }
   HyPerLayer *preLayer = connectionData->getPre();

   initializeDelays();
   int maxDelay     = maxDelaySteps();
   int allowedDelay = preLayer->increaseDelayLevels(maxDelay);
   if (allowedDelay < maxDelay) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: attempt to set delay to %d, but the maximum "
               "allowed delay is %d.  Exiting\n",
               getDescription_c(),
               maxDelay,
               allowedDelay);
      }
      MPI_Barrier(parent->getCommunicator()->globalCommunicator());
      exit(EXIT_FAILURE);
   }

   return Response::SUCCESS;
}

void ArborList::initializeDelays() {
   assert(!parent->parameters()->presentAndNotBeenRead(this->getName(), "numAxonalArbors"));
   mDelay.resize(getNumAxonalArbors());

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
      else if (mNumDelays == getNumAxonalArbors()) {
         setDelay(arborId, mDelaysParams[arborId]);
      }
      else {
         Fatal().printf(
               "Delay must be either a single value or the same length "
               "as the number of arbors\n");
      }
   }
}

void ArborList::setDelay(int arborId, double delay) {
   assert(arborId >= 0 && arborId < getNumAxonalArbors());
   int intDelay = (int)std::nearbyint(delay / parent->getDeltaTime());
   if (std::fmod(delay, parent->getDeltaTime()) != 0) {
      double actualDelay = intDelay * parent->getDeltaTime();
      WarnLog() << getName() << ": A delay of " << delay << " will be rounded to " << actualDelay
                << "\n";
   }
   mDelay[arborId] = intDelay;
}

int ArborList::maxDelaySteps() {
   int maxDelay        = 0;
   int const numArbors = getNumAxonalArbors();
   for (auto &d : mDelay) {
      if (d > maxDelay) {
         maxDelay = d;
      }
   }
   return maxDelay;
}

} // namespace PV
