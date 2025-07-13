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

PlasticCloneConn::PlasticCloneConn(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

PlasticCloneConn::~PlasticCloneConn() {}

void PlasticCloneConn::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   CloneConn::initialize(params, defaults, comm);
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
   std::string const &originalConnName = originalConnNameParam->getLinkedObjectName();

   auto *originalUpdater = objectTable->findObject<HebbianUpdater>(originalConnName);
   FatalIf(
         originalUpdater == nullptr,
         "%s specifies originalConnName \"%s\", but this connection does not have a "
         "Hebbian updater.\n",
         getDescription_c(),
         originalConnName.c_str());
   // Do we need to handle PlasticClones of PlasticClones? Right now, this won't handle that case.
   originalUpdater->addClone(connectionData);

   return Response::SUCCESS;
}

} // end namespace PV
