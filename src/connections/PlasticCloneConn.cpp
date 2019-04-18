/* PlasticCloneConn.cpp
 *
 * Created on: May 23, 2011
 *     Author: peteschultz
 */

#include "PlasticCloneConn.hpp"
#include "columns/HyPerCol.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "weightupdaters/HebbianUpdater.hpp"

namespace PV {

PlasticCloneConn::PlasticCloneConn() {}

PlasticCloneConn::PlasticCloneConn(const char *name, HyPerCol *hc) { initialize(name, hc); }

PlasticCloneConn::~PlasticCloneConn() {}

int PlasticCloneConn::initialize(const char *name, HyPerCol *hc) {
   int status = CloneConn::initialize(name, hc);
   return status;
}

Response::Status
PlasticCloneConn::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = CloneConn::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *originalConnNameParam = getComponentByType<OriginalConnNameParam>();
   FatalIf(
         originalConnNameParam == nullptr,
         "%s requires an OriginalConnNameParam component.\n",
         getDescription_c());
   auto *originalConnName = originalConnNameParam->getOriginalConnName();
   auto *originalConn     = message->lookup<HyPerConn>(std::string(originalConnName));
   pvAssert(originalConn); // CloneConn::communicateInitInfo should have failed if this fails.
   auto *originalUpdater = originalConn->getComponentByType<HebbianUpdater>();
   FatalIf(
         originalUpdater == nullptr,
         "%s specifies %s as its original connection, but this connection does not have a "
         "Hebbian updater.\n",
         getDescription_c(),
         originalConn->getDescription_c());
   // Do we need to handle PlasticClones of PlasticClones? Right now, this won't handle that case.
   auto *connectionData = getComponentByType<ConnectionData>();
   pvAssert(connectionData); // BaseConnection creates this component
   pvAssert(connectionData->getInitInfoCommunicatedFlag()); // Set while CloneConn communicates
   // CloneConn creates this component, and
   // CloneConn::CommunicateInitInfo shouldn't return until its components have communicated.
   originalUpdater->addClone(connectionData);

   return Response::SUCCESS;
}

} // end namespace PV
