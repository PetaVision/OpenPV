/* PlasticCloneConn.cpp
 *
 * Created on: May 23, 2011
 *     Author: peteschultz
 */

#include "PlasticCloneConn.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "weightupdaters/HebbianUpdater.hpp"

namespace PV {

PlasticCloneConn::PlasticCloneConn() {}

PlasticCloneConn::PlasticCloneConn(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

PlasticCloneConn::~PlasticCloneConn() {}

void PlasticCloneConn::initialize(const char *name, PVParams *params, Communicator const *comm) {
   CloneConn::initialize(name, params, comm);
}

Response::Status
PlasticCloneConn::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = CloneConn::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *objectTable = message->mObjectTable;

   auto *connectionData = objectTable->findObject<ConnectionData>(getName());
   FatalIf(
         connectionData == nullptr,
         "%s could not find a ConnectionData component.\n",
         getDescription_c());
   if (!connectionData->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   auto *originalConnNameParam = objectTable->findObject<OriginalConnNameParam>(getName());
   FatalIf(
         originalConnNameParam == nullptr,
         "%s requires an OriginalConnNameParam component.\n",
         getDescription_c());
   if (!originalConnNameParam->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   char const *originalConnName = originalConnNameParam->getLinkedObjectName();

   auto *originalUpdater = objectTable->findObject<HebbianUpdater>(originalConnName);
   FatalIf(
         originalUpdater == nullptr,
         "%s specifies originalConnName \"%s\", but this connection does not have a "
         "Hebbian updater.\n",
         getDescription_c(),
         originalConnName);
   // Do we need to handle PlasticClones of PlasticClones? Right now, this won't handle that case.
   originalUpdater->addClone(connectionData);

   return Response::SUCCESS;
}

} // end namespace PV
