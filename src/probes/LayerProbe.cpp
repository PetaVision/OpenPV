/*
 * LayerProbe.cpp
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#include "LayerProbe.hpp"
#include "../layers/HyPerLayer.hpp"

namespace PV {

LayerProbe::LayerProbe()
{
   initialize_base();
   // Derived classes of LayerProbe should call LayerProbe::initialize themselves.
}

/**
 * @filename
 */
LayerProbe::LayerProbe(const char * probeName, HyPerCol * hc)
{
   initialize_base();
   initialize(probeName, hc);
}

LayerProbe::~LayerProbe()
{
}

int LayerProbe::initialize_base() {
   targetLayer = NULL;
   return PV_SUCCESS;
}

/**
 * @filename
 * @layer
 */
int LayerProbe::initialize(const char * probeName, HyPerCol * hc)
{
   int status = BaseProbe::initialize(probeName, hc);
   return status;
}

void LayerProbe::ioParam_targetName(enum ParamsIOFlag ioFlag) {
   //targetLayer is a legacy parameter, so here, it's not required
   parent->parameters()->ioParamString(ioFlag, name, "targetLayer", &targetName, NULL/*default*/, false/*warnIfAbsent*/);
   //But if it isn't defined, targetName must be, which BaseProbe checks for
   if(targetName == NULL){
      BaseProbe::ioParam_targetName(ioFlag);
   }
}

int LayerProbe::communicateInitInfo() {
   BaseProbe::communicateInitInfo();
   //Set target layer
   int status = setTargetLayer(targetName);
   //Add to layer
   if (status == PV_SUCCESS) {
      targetLayer->insertProbe(this);
   }
   return status;
}

int LayerProbe::setTargetLayer(const char * layerName) {
   targetLayer = parent->getLayerFromName(layerName);
   if (targetLayer==NULL) {
      if (parent->columnId()==0) {
         pvErrorNoExit().printf("%s: targetLayer \"%s\" is not a layer in the column.\n",
               getDescription_c(), layerName);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

bool LayerProbe::needRecalc(double timevalue) {
   return this->getLastUpdateTime() < targetLayer->getLastUpdateTime();
}

double LayerProbe::referenceUpdateTime() const {
   return targetLayer->getLastUpdateTime();
}

} // namespace PV
