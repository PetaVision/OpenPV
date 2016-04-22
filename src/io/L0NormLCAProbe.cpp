/*
 * L0NormLCAProbe.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#include "L0NormLCAProbe.hpp"
#include "../layers/HyPerLCALayer.hpp"

namespace PV {

L0NormLCAProbe::L0NormLCAProbe(const char * probeName, HyPerCol * hc) {
   initialize_base();
   initL0NormLCAProbe(probeName, hc);
}

L0NormLCAProbe::L0NormLCAProbe() {
   initialize_base();
}

int L0NormLCAProbe::communicateInitInfo() {
   int status = L0NormProbe::communicateInitInfo();
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
      pvdata_t vThresh= targetLCALayer->getVThresh();
      coefficient = vThresh*vThresh/2.0f;
   }
   return status;
}

BaseObject * createL0NormLCAProbe(char const * name, HyPerCol * hc) {
   return hc ? new L0NormLCAProbe(name, hc) : NULL;
}

} /* namespace PV */
