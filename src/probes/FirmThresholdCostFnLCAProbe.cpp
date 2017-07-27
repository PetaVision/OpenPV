/*
 * FirmThresholdCostFnLCAProbe.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#include "FirmThresholdCostFnLCAProbe.hpp"
#include "../layers/HyPerLCALayer.hpp"

namespace PV {

FirmThresholdCostFnLCAProbe::FirmThresholdCostFnLCAProbe(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

FirmThresholdCostFnLCAProbe::FirmThresholdCostFnLCAProbe() { initialize_base(); }

int FirmThresholdCostFnLCAProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = FirmThresholdCostFnProbe::communicateInitInfo(message);
   assert(targetLayer);
   HyPerLCALayer *targetLCALayer = dynamic_cast<HyPerLCALayer *>(targetLayer);
   if (targetLCALayer == nullptr) {
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
      coefficient = targetLCALayer->getVThresh();
   }
   return status;
}

} /* namespace PV */
