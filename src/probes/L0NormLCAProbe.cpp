/*
 * L0NormLCAProbe.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#include "L0NormLCAProbe.hpp"
#include "../layers/HyPerLCALayer.hpp"

namespace PV {

L0NormLCAProbe::L0NormLCAProbe(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

L0NormLCAProbe::L0NormLCAProbe() { initialize_base(); }

int L0NormLCAProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = L0NormProbe::communicateInitInfo(message);
   assert(targetLayer);
   HyPerLCALayer *targetLCALayer = dynamic_cast<HyPerLCALayer *>(targetLayer);
   if (targetLCALayer == NULL) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: targetLayer \"%s\" is not an LCA layer.\n",
               getDescription_c(),
               getTargetName());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (targetLCALayer->layerListsVerticesInParams() == true) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: LCAProbes require targetLayer \"%s\" to use "
               "VThresh etc. instead of "
               "verticesV/verticesV.\n",
               getDescription_c(),
               getTargetName());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (status == PV_SUCCESS) {
      float vThresh = targetLCALayer->getVThresh();
      coefficient   = vThresh * vThresh / 2.0f;
   }
   return status;
}

} /* namespace PV */
