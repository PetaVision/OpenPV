/*
 * L1NormLCAProbe.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#include "L1NormLCAProbe.hpp"
#include "../layers/HyPerLCALayer.hpp"

namespace PV {

L1NormLCAProbe::L1NormLCAProbe(const char * probeName, HyPerCol * hc) {
   initialize_base();
   initL1NormLCAProbe(probeName, hc);
}

L1NormLCAProbe::L1NormLCAProbe() {
   initialize_base();
}

int L1NormLCAProbe::communicateInitInfo() {
   int status = L1NormProbe::communicateInitInfo();
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

BaseObject * createL1NormLCAProbe(char const * name, HyPerCol * hc) {
   return hc ? new L1NormLCAProbe(name, hc) : NULL;
}

} /* namespace PV */
