/*
 * ConnectionData.cpp
 *
 *  Created on: Nov 17, 2017
 *      Author: pschultz
 */

#include "ConnectionData.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

ConnectionData::ConnectionData(char const *name, HyPerCol *hc) { initialize(name, hc); }

ConnectionData::ConnectionData() {}

ConnectionData::~ConnectionData() {
   free(mPreLayerName);
   free(mPostLayerName);
}

int ConnectionData::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

void ConnectionData::setObjectType() { mObjectType = "ConnectionData"; }

int ConnectionData::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_preLayerName(ioFlag);
   ioParam_postLayerName(ioFlag);
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

Response::Status
ConnectionData::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (getPreLayerName() == nullptr and getPostLayerName() == nullptr) {
      std::string preLayerNameString, postLayerNameString;
      inferPreAndPostFromConnName(
            getName(),
            parent->getCommunicator()->globalCommRank(),
            preLayerNameString,
            postLayerNameString);
      mPreLayerName  = strdup(preLayerNameString.c_str());
      mPostLayerName = strdup(postLayerNameString.c_str());
   }
   MPI_Barrier(this->parent->getCommunicator()->globalCommunicator());
   if (getPreLayerName() == nullptr or getPostLayerName() == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: Unable to determine pre- and post-layer names. Exiting.\n", getDescription_c());
      }
      exit(EXIT_FAILURE);
   }

   auto hierarchy = message->mHierarchy;
   ObjectMapComponent *objectMapComponent =
         mapLookupByType<ObjectMapComponent>(hierarchy, getDescription());
   FatalIf(
         objectMapComponent == nullptr,
         "CommunicateInitInfo called for %s with no ObjectMapComponent object.\n",
         getDescription_c());

   bool failed = false;
   mPre        = objectMapComponent->lookup<HyPerLayer>(std::string(getPreLayerName()));
   if (getPre() == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: preLayerName \"%s\" does not correspond to a layer in the column.\n",
               getDescription_c(),
               getPreLayerName());
      }
      failed = true;
   }

   mPost = objectMapComponent->lookup<HyPerLayer>(std::string(getPostLayerName()));
   if (getPost() == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: postLayerName \"%s\" does not correspond to a layer in the column.\n",
               getDescription_c(),
               getPostLayerName());
      }
      failed = true;
   }
   MPI_Barrier(parent->getCommunicator()->globalCommunicator());
   if (failed) {
      exit(EXIT_FAILURE);
   }

   return Response::SUCCESS;
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

} // namespace PV
