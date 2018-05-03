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
LayerProbe::LayerProbe(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
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
int LayerProbe::initialize(const char *name, HyPerCol *hc) {
   int status = BaseProbe::initialize(name, hc);
   return status;
}

void LayerProbe::ioParam_targetName(enum ParamsIOFlag ioFlag) {
   // targetLayer is a legacy parameter, so here, it's not required
   parent->parameters()->ioParamString(
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
   targetLayer = message->lookup<HyPerLayer>(std::string(targetName));
   if (targetLayer == NULL) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: targetLayer \"%s\" is not a layer in the column.\n",
               getDescription_c(),
               targetName);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   // Add to layer
   targetLayer->insertProbe(this);
   return Response::SUCCESS;
}

bool LayerProbe::needRecalc(double timevalue) {
   return this->getLastUpdateTime() < targetLayer->getLastUpdateTime();
}

double LayerProbe::referenceUpdateTime() const { return targetLayer->getLastUpdateTime(); }

} // namespace PV
