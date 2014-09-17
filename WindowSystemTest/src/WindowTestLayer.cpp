/*
 * WindowTestLayer.cpp
 *
 *  Created on: Sep 27, 2011
 *      Author: slundquist
 */

#include "WindowTestLayer.hpp"
#include <utils/conversions.h>

namespace PV {

WindowTestLayer::WindowTestLayer(const char * name, HyPerCol * hc){
   initialize_base();
   initialize(name, hc);
}

int WindowTestLayer::initialize_base()
{
   //windowLayerName = NULL;
   //windowLayer = NULL;
   return PV_SUCCESS;
}

int WindowTestLayer::initialize(const char * name, HyPerCol * hc)
{
   ANNLayer::initialize(name, hc);
   //PVParams * params = parent->parameters();
   //windowLayerName = params->stringValue(name, "windowLayerName", NULL);
   //if (windowLayerName == NULL){
   //   fprintf(stderr, "WindowTestLayer \"%s\" error: windowLayerName must be specified.\n", name);
   //   abort();
   //}
   return PV_SUCCESS;
}

int WindowTestLayer::communicateInitInfo() {
   int status = ANNLayer::communicateInitInfo();
   //HyPerLayer * origLayer = parent->getLayerFromName(windowLayerName);
   //if (origLayer ==NULL) {
   //   fprintf(stderr, "WindowTestLayer \"%s\" error: windowLayerName \"%s\" is not a layer in the HyPerCol.\n", name, windowLayerName);
   //   abort();
   //}
   ////For now, the window information must come from HyPerLCALayer
   //windowLayer = dynamic_cast<HyPerLCALayer *>(origLayer);
   //if (windowLayer ==NULL) {
   //   fprintf(stderr, "WindowTestLayer \"%s\" error: windowLayerName \"%s\" must be a HyPerLCALayer.\n", name, windowLayerName);
   //   abort();
   //}
   return status;
}

int WindowTestLayer::allocateDataStructures() {
   int status = ANNLayer::allocateDataStructures();
   if (status==PV_SUCCESS) {
      setActivitytoOne();
   }
   return status;
}

// set activity to global x/y/f position, using position in border/margin as required
int WindowTestLayer::setActivitytoOne(){
   //Make sure window layer is the same size as current layer
   //const PVLayerLoc * windowLoc = windowLayer->getLayerLoc();
   //const PVLayerLoc * thisLoc = this->getLayerLoc();
   //if(windowLoc->nx != thisLoc->nx || windowLoc->ny != thisLoc->ny || windowLoc->nf != thisLoc->nf || windowLoc->nb != thisLoc->nb){
   //   fprintf(stderr, "WindowTestLayer \"%s\" error: Size (including margins) must equal to the window layer.\n", name);
   //   abort();
   //}
   for (int kLocalExt = 0; kLocalExt < clayer->numExtended; kLocalExt++){
      //int kWindow = layerIndexExt(kLocalExt, thisLoc, windowLoc);
      //int kxGlobalExt = kxPos(kWindow, windowLoc->nx + 2*windowLoc->nb, windowLoc->ny + 2*windowLoc->nb, windowLoc->nf) + windowLoc->kx0;
      //int kyGlobalExt = kyPos(kWindow, windowLoc->nx + 2*windowLoc->nb, windowLoc->ny + 2*windowLoc->nb, windowLoc->nf) + windowLoc->ky0;
      //int kxGlobalExt = kxPos(kLocalExt, windowLoc->nx + 2*windowLoc->nb, windowLoc->ny + 2*windowLoc->nb, windowLoc->nf) + windowLoc->kx0;
      //int kyGlobalExt = kyPos(kLocalExt, windowLoc->nx + 2*windowLoc->nb, windowLoc->ny + 2*windowLoc->nb, windowLoc->nf) + windowLoc->ky0;
      //Get window from windowLayer
      //int windowId = windowLayer->calcWindow(kxGlobalExt, kyGlobalExt);
      clayer->activity->data[kLocalExt] = 1;
   }
   return PV_SUCCESS;
}


int WindowTestLayer::updateState(double timed, double dt)
{
   //updateV();
   //setActivity();
   //resetGSynBuffers();
   //updateActiveIndices();

   return PV_SUCCESS;
}

int WindowTestLayer::publish(InterColComm* comm, double timed)
{
   setActivitytoOne();
   int status = comm->publish(this, clayer->activity);
   return status;

   //return HyPerLayer::publish(comm, time);
}

} /* namespace PV */
