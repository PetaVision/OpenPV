/*
 * ConnectionData.cpp
 *
 *  Created on: Nov 17, 2017
 *      Author: pschultz
 */

#include "ConnectionData.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"

namespace PV {

ConnectionData::ConnectionData(char const *name, HyPerCol *hc) { initialize(name, hc); }

ConnectionData::ConnectionData() {}

ConnectionData::~ConnectionData() {}

int ConnectionData::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

int ConnectionData::setDescription() {
   description = "ConnectionData \"";
   description += name;
   description += "\"";
   return PV_SUCCESS;
}

int ConnectionData::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_preLayerName(ioFlag);
   ioParam_postLayerName(ioFlag);
   ioParam_numAxonalArbors(ioFlag);
   ioParam_delay(ioFlag);
   return PV_SUCCESS;
}

void ConnectionData::ioParam_preLayerName(enum ParamsIOFlag ioFlag) {
   this->parent->parameters()->ioParamString(
         ioFlag, this->getName(), "preLayerName", &mPreLayerName, NULL, false /*warnIfAbsent*/);
}

void ConnectionData::ioParam_postLayerName(enum ParamsIOFlag ioFlag) {
   this->parent->parameters()->ioParamString(
         ioFlag, this->getName(), "postLayerName", &mPostLayerName, NULL, false /*warnIfAbsent*/);
}

void ConnectionData::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, this->getName(), "numAxonalArbors", &mNumAxonalArbors, mNumAxonalArbors);
   if (ioFlag == PARAMS_IO_READ) {
      if (getNumAxonalArbors() <= 0 && parent->columnId() == 0) {
         WarnLog().printf(
               "Connection %s: Variable numAxonalArbors is set to 0. "
               "No connections will be made.\n",
               this->getName());
      }
   }
}

void ConnectionData::ioParam_delay(enum ParamsIOFlag ioFlag) {
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

int ConnectionData::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = PV_SUCCESS;

   if (getPreLayerName() == nullptr and getPostLayerName() == nullptr) {
      handleMissingPreAndPostLayerNames();
   }
   MPI_Barrier(this->parent->getCommunicator()->communicator());
   if (getPreLayerName() == nullptr or getPostLayerName() == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: Unable to determine pre- and post-layer names. Exiting.\n", getDescription_c());
      }
      exit(EXIT_FAILURE);
   }

   // TODO: Use MapLookupByType
   ObjectMapComponent *objectMapComponent = nullptr;
   auto hierarchy                         = message->mHierarchy;
   for (auto &objpair : hierarchy) {
      auto *obj           = objpair.second;
      auto connectionData = dynamic_cast<ObjectMapComponent *>(obj);
      if (connectionData != nullptr) {
         FatalIf(
               objectMapComponent != nullptr,
               "CommunicateInitInfo called for %s with more than one ObjectMapComponent object.\n",
               getDescription_c());
         objectMapComponent = connectionData;
      }
   }
   FatalIf(
         objectMapComponent == nullptr,
         "CommunicateInitInfo called for %s with no ObjectMapComponent object.\n",
         getDescription_c());

   mPre = objectMapComponent->lookup<HyPerLayer>(std::string(getPreLayerName()));
   if (getPre() == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: preLayerName \"%s\" does not correspond to a layer in the column.\n",
               getDescription_c(),
               getPreLayerName());
      }
      status = PV_FAILURE;
   }

   mPost = objectMapComponent->lookup<HyPerLayer>(std::string(getPostLayerName()));
   if (getPost() == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: postLayerName \"%s\" does not correspond to a layer in the column.\n",
               getDescription_c(),
               getPostLayerName());
      }
      status = PV_FAILURE;
   }
   MPI_Barrier(parent->getCommunicator()->communicator());
   if (status != PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }

   initializeDelays();
   int maxDelay     = maxDelaySteps();
   int allowedDelay = getPre()->increaseDelayLevels(maxDelay);
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

   return status;
}

void ConnectionData::handleMissingPreAndPostLayerNames() {
   std::string preLayerNameString, postLayerNameString;
   inferPreAndPostFromConnName(
         getName(),
         parent->getCommunicator()->globalCommRank(),
         preLayerNameString,
         postLayerNameString);
   mPreLayerName  = strdup(preLayerNameString.c_str());
   mPostLayerName = strdup(postLayerNameString.c_str());
}

void ConnectionData::inferPreAndPostFromConnName(
      const char *name,
      int rank,
      std::string &preLayerNameString,
      std::string &postLayerNameString) {
   pvAssert(name);
   preLayerNameString.clear();
   postLayerNameString.clear();
   std::string nameString(name);
   auto locto = nameString.find("To");
   if (locto == std::string::npos) {
      if (rank == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf("Unable to infer pre and post from connection name \"%s\".\n", name);
         errorMessage.printf(
               "The connection name must have the form \"AbcToXyz\", to infer the names,\n");
         errorMessage.printf("but the string \"To\" does not appear.\n");
         return;
      }
   }
   auto secondto = nameString.find("To", locto + 1);
   if (secondto != std::string::npos) {
      if (rank == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf("Unable to infer pre and post from connection name \"%s\":\n", name);
         errorMessage.printf("The string \"To\" cannot appear in the name more than once.\n");
      }
   }
   preLayerNameString.append(nameString.substr(0, locto));
   postLayerNameString.append(nameString.substr(locto + 2, std::string::npos));
}

void ConnectionData::initializeDelays() {
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

void ConnectionData::setDelay(int arborId, double delay) {
   assert(arborId >= 0 && arborId < getNumAxonalArbors());
   int intDelay = (int)std::nearbyint(delay / parent->getDeltaTime());
   if (std::fmod(delay, parent->getDeltaTime()) != 0) {
      double actualDelay = intDelay * parent->getDeltaTime();
      WarnLog() << getName() << ": A delay of " << delay << " will be rounded to " << actualDelay
                << "\n";
   }
   mDelay[arborId] = intDelay;
}

int ConnectionData::maxDelaySteps() {
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
