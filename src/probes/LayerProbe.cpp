/*
 * LayerProbe.cpp
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#include "LayerProbe.hpp"
#include "../layers/HyPerLayer.hpp"

namespace PV {

LayerProbe::LayerProbe() {
   initialize_base();
   // Derived classes of LayerProbe should call LayerProbe::initialize
   // themselves.
}

/**
 * @filename
 */
LayerProbe::LayerProbe(const char *name, PVParams *params, Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
}

LayerProbe::~LayerProbe() {}

int LayerProbe::initialize_base() {
   targetLayer = NULL;
   return PV_SUCCESS;
}

/**
 * @filename
 * @layer
 */
void LayerProbe::initialize(const char *name, PVParams *params, Communicator *comm) {
   BaseProbe::initialize(name, params, comm);
}

void LayerProbe::ioParam_targetName(enum ParamsIOFlag ioFlag) {
   // targetLayer is a legacy parameter, so here, it's not required
   parameters()->ioParamString(
         ioFlag, name, "targetLayer", &targetName, NULL /*default*/, false /*warnIfAbsent*/);
   // But if it isn't defined, targetName must be, which BaseProbe checks for
   if (targetName == NULL) {
      BaseProbe::ioParam_targetName(ioFlag);
   }
}

Response::Status
LayerProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseProbe::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   // Set target layer
   targetLayer = message->mHierarchy->lookupByName<HyPerLayer>(std::string(targetName));
   if (targetLayer == NULL) {
      if (mCommunicator->commRank() == 0) {
         ErrorLog().printf(
               "%s: targetLayer \"%s\" is not a layer in the column.\n",
               getDescription_c(),
               targetName);
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }

   // Add to layer
   targetLayer->insertProbe(this);
   return Response::SUCCESS;
}

bool LayerProbe::needRecalc(double timevalue) {
   auto *updateController = targetLayer->getComponentByType<LayerUpdateController>();
   pvAssert(updateController);
   return this->getLastUpdateTime() < updateController->getLastUpdateTime();
}

double LayerProbe::referenceUpdateTime(double simTime) const {
   auto *updateController = targetLayer->getComponentByType<LayerUpdateController>();
   pvAssert(updateController);
   return updateController->getLastUpdateTime();
}

} // namespace PV
