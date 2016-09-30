/*
 * L1NormLCAProbe.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#include "L1NormLCAProbe.hpp"
#include "../layers/HyPerLCALayer.hpp"

namespace PV {

L1NormLCAProbe::L1NormLCAProbe(const char *probeName, HyPerCol *hc) {
   initialize_base();
   initL1NormLCAProbe(probeName, hc);
}

L1NormLCAProbe::L1NormLCAProbe() { initialize_base(); }

int L1NormLCAProbe::communicateInitInfo() {
   int status = L1NormProbe::communicateInitInfo();
   assert(targetLayer);
   HyPerLCALayer *targetLCALayer = dynamic_cast<HyPerLCALayer *>(targetLayer);
   if (targetLCALayer == NULL) {
      if (parent->columnId() == 0) {
         pvErrorNoExit().printf(
               "%s: targetLayer \"%s\" is not an LCA layer.\n",
               getDescription_c(),
               getTargetName());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (targetLCALayer->layerListsVerticesInParams() == true) {
      if (parent->columnId() == 0) {
         pvErrorNoExit().printf(
               "%s: LCAProbes require targetLayer \"%s\" to use VThresh etc. instead of "
               "verticesV/verticesV.\n",
               getDescription_c(),
               getTargetName());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (status == PV_SUCCESS) {
      coefficient = targetLCALayer->getVThresh();
   }
   return status;
}

} /* namespace PV */
