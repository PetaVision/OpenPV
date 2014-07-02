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
   parent->ioParamString(ioFlag, name, "targetLayer", &targetName, NULL/*default*/, false/*warnIfAbsent*/);
   //But if it isn't defined, targetName must be, which BaseProbe checks for
   if(targetName == NULL){
      BaseProbe::ioParam_targetName(ioFlag);
   }
}

int LayerProbe::communicateInitInfo() {
   BaseProbe::communicateInitInfo();
   //Set target layer
   int status = setTargetLayer(targetName);
   owner = targetName;
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
         fprintf(stderr, "%s \"%s\" error: targetLayer \"%s\" is not a layer in the column.\n",
               parent->parameters()->groupKeywordFromName(name), name, layerName);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

} // namespace PV
