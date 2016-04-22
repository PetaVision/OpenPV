/*
 * FirmThresholdCostFnLCAProbe.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#include "FirmThresholdCostFnLCAProbe.hpp"
#include "../layers/HyPerLCALayer.hpp"

namespace PV {

FirmThresholdCostFnLCAProbe::FirmThresholdCostFnLCAProbe(const char * probeName, HyPerCol * hc) {
   initialize_base();
   initFirmThresholdCostFnLCAProbe(probeName, hc);
}

FirmThresholdCostFnLCAProbe::FirmThresholdCostFnLCAProbe() {
   initialize_base();
}

int FirmThresholdCostFnLCAProbe::communicateInitInfo() {
   int status = FirmThresholdCostFnProbe::communicateInitInfo();
   assert(targetLayer);
   HyPerLCALayer * targetLCALayer = dynamic_cast<HyPerLCALayer *>(targetLayer);
   if (targetLCALayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: targetLayer \"%s\" is not an LCA layer.\n",
               getKeyword(), getName(), getTargetName());
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (status == PV_SUCCESS) {
      coefficient = targetLCALayer->getVThresh();
   }
   return status;
}

BaseObject * createFirmThresholdCostFnLCAProbe(char const * name, HyPerCol * hc) {
   return hc ? new FirmThresholdCostFnLCAProbe(name, hc) : NULL;
}

} /* namespace PV */
