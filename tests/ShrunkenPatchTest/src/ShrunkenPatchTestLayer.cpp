/*
 * ShrunkenPatchTestLayer.cpp
 *
 *  Created on: Sep 27, 2011
 *      Author: gkenyon
 */

#include "ShrunkenPatchTestLayer.hpp"
#include <utils/conversions.h>

namespace PV {

ShrunkenPatchTestLayer::ShrunkenPatchTestLayer(const char * name, HyPerCol * hc) : ANNLayer() {
   // ShrunkenPatchTestLayer has no member variables to initialize in initialize_base()
   initialize(name, hc);
}

// set V to global x/y/f position
int ShrunkenPatchTestLayer::setVtoGlobalPos(){
   for (int kLocal = 0; kLocal < clayer->numNeurons; kLocal++){
      int kGlobal = globalIndexFromLocal(kLocal, clayer->loc);
      int kxGlobal = kxPos(kGlobal, clayer->loc.nxGlobal, clayer->loc.nyGlobal, clayer->loc.nf);
      float xScaleLog2 = clayer->xScale;
      float x0 = xOriginGlobal(xScaleLog2);
      float dx = deltaX(xScaleLog2);
      float x_global_pos = (x0 + dx * kxGlobal);
      clayer->V[kLocal] = x_global_pos;
   }
   return PV_SUCCESS;
}


// set activity to global x/y/f position, using position in border/margin as required
int ShrunkenPatchTestLayer::setActivitytoGlobalPos(){
   for (int kLocalExt = 0; kLocalExt < clayer->numExtended; kLocalExt++){
      int kxLocalExt = kxPos(kLocalExt, clayer->loc.nx + clayer->loc.halo.lt + clayer->loc.halo.rt, clayer->loc.ny + clayer->loc.halo.dn + clayer->loc.halo.up, clayer->loc.nf) - clayer->loc.halo.lt;
      int kxGlobalExt = kxLocalExt + clayer->loc.kx0;
      float xScaleLog2 = clayer->xScale;
      float x0 = xOriginGlobal(xScaleLog2);
      float dx = deltaX(xScaleLog2);
      float x_global_pos = (x0 + dx * kxGlobalExt);
      int kyLocalExt = kyPos(kLocalExt, clayer->loc.nx + clayer->loc.halo.lt + clayer->loc.halo.rt, clayer->loc.ny + clayer->loc.halo.dn + clayer->loc.halo.up, clayer->loc.nf) - clayer->loc.halo.up;
      int kyGlobalExt = kyLocalExt + clayer->loc.ky0;

      bool x_in_local_interior = kxLocalExt >= 0 && kxLocalExt < clayer->loc.nx;
      bool y_in_local_interior = kyLocalExt >= 0 && kyLocalExt < clayer->loc.ny;

      bool x_in_global_boundary = kxGlobalExt < 0 || kxGlobalExt >= clayer->loc.nxGlobal;
      bool y_in_global_boundary = kyGlobalExt < 0 || kyGlobalExt >= clayer->loc.nyGlobal;

      if( ( x_in_global_boundary || x_in_local_interior ) && ( y_in_global_boundary || y_in_local_interior )  ) {
         clayer->activity->data[kLocalExt] = x_global_pos;
      }
   }
   return PV_SUCCESS;
}


int ShrunkenPatchTestLayer::initialize(const char * name, HyPerCol * hc){
   ANNLayer::initialize(name, hc);

   return PV_SUCCESS;
}

int ShrunkenPatchTestLayer::allocateDataStructures() {
   int status = ANNLayer::allocateDataStructures();
   if (status==PV_SUCCESS) {
      setVtoGlobalPos();
      setActivitytoGlobalPos();
   }
   return status;
}

int ShrunkenPatchTestLayer::updateState(double timed, double dt)
{
   //updateV();
   //setActivity();
   //resetGSynBuffers();
   //updateActiveIndices();

   return PV_SUCCESS;
}

int ShrunkenPatchTestLayer::publish(InterColComm* comm, double timed)
{
   setActivitytoGlobalPos();
   int status = comm->publish(this, clayer->activity);
   return status;

   //return HyPerLayer::publish(comm, time);
}

BaseObject * createShrunkenPatchTestLayer(char const * name, HyPerCol * hc) {
   return hc ? new ShrunkenPatchTestLayer(name, hc) : NULL;
}

} /* namespace PV */
