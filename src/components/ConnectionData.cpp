/*
 * ConnectionData.cpp
 *
 *  Created on: Nov 17, 2017
 *      Author: pschultz
 */

#include "ConnectionData.hpp"
#include "components/LayerGeometry.hpp"
#include "observerpattern/ObserverTable.hpp"
#include "utils/PVAssert.hpp"

namespace PV {

ConnectionData::ConnectionData(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

ConnectionData::ConnectionData() {}

ConnectionData::~ConnectionData() {}

void ConnectionData::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   BaseObject::initialize(paramsIO, comm);
}

void ConnectionData::setObjectType() { mObjectType = "ConnectionData"; }

int ConnectionData::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_preLayerName(ioSwitch);
   ioParam_postLayerName(ioSwitch);
   return PV_SUCCESS;
}

void ConnectionData::ioParam_preLayerName(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "preLayerName", &mPreLayerName, false /*warnIfAbsentFlag*/);
}

void ConnectionData::ioParam_postLayerName(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "postLayerName", &mPostLayerName, false /*warnIfAbsentFlag*/);
}

Response::Status
ConnectionData::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (getPreLayerName().empty() and getPostLayerName().empty()) {
      inferPreAndPostFromConnName(
            getName(), mCommunicator->globalCommRank(), mPreLayerName, mPostLayerName);
   }
   FatalIf(
         getPreLayerName().empty() or getPostLayerName().empty(),
         "%s: Unable to determine pre- and post-layer names. Exiting.\n", getDescription_c());

   auto objectTable = message->mObjectTable;

   bool failed = false;
   mPre        = objectTable->findObject<HyPerLayer>(getPreLayerName());
   if (getPre() == nullptr) {
      if (mCommunicator->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: preLayerName \"%s\" does not correspond to a layer in the column.\n",
               getDescription_c(),
               getPreLayerName().c_str());
      }
      failed = true;
   }

   mPost = objectTable->findObject<HyPerLayer>(getPostLayerName());
   if (getPost() == nullptr) {
      if (mCommunicator->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: postLayerName \"%s\" does not correspond to a layer in the column.\n",
               getDescription_c(),
               getPostLayerName());
      }
      failed = true;
   }
   MPI_Barrier(mCommunicator->globalCommunicator());
   if (failed) {
      std::exit(EXIT_FAILURE);
   }
   if (!mPre->getInitInfoCommunicatedFlag() or !mPost->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   auto *preGeom = objectTable->findObject<LayerGeometry>(getPreLayerName());
   pvAssert(preGeom);
   mPreIsBroadcast = preGeom->getBroadcastFlag();

   auto *postGeom = objectTable->findObject<LayerGeometry>(getPostLayerName());
   pvAssert(postGeom);
   mPostIsBroadcast = postGeom->getBroadcastFlag();

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
