/*
 * SigmoidLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "HyPerLayer.hpp"
#include "SigmoidLayer.hpp"
#include <stdio.h>

#include "../include/default_params.h"

// CloneLayer can be used to implement Sigmoid junctions
namespace PV {
SigmoidLayer::SigmoidLayer() {
   initialize_base();
}

SigmoidLayer::SigmoidLayer(const char * name, HyPerCol * hc, LIF * originalLayer) {
   initialize_base();
   initialize(name, hc, originalLayer);
}

SigmoidLayer::~SigmoidLayer()
{
    clayer->V = NULL;
}

int SigmoidLayer::initialize_base() {
   sourceLayer = NULL;
   return PV_SUCCESS;
}

int SigmoidLayer::initialize(const char * name, HyPerCol * hc, LIF * clone) {
   int status_init = HyPerLayer::initialize(name, hc, MAX_CHANNELS);

   V0 = parent->parameters()->value(name, "Vrest", V_REST);
   Vth = parent->parameters()->value(name,"VthRest",VTH_REST);
   InverseFlag = parent->parameters()->value(name,"InverseFlag",INVERSEFLAG);
   SigmoidFlag = parent->parameters()->value(name,"SigmoidFlag",SIGMOIDFLAG);
   SigmoidAlpha = parent->parameters()->value(name,"SigmoidAlpha",SIGMOIDALPHA);

   if (parent->columnId()==0) {
      if(InverseFlag)   fprintf(stdout,"SigmoidLayer: Inverse flag is set\n");
      if(SigmoidFlag)   fprintf(stdout,"SigmoidLayer: True Sigmoid flag is set\n");
   }

   this->spikingFlag = false;
   sourceLayer = clone;
   free(clayer->V);
   clayer->V = sourceLayer->getV();

   // don't need conductance channels
   freeChannels();

   return status_init;
}

int SigmoidLayer::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtended);
   return 0;
}

int SigmoidLayer::updateState(double timef, double dt) {
   int status;
   status = updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), 0, NULL, Vth, V0, SigmoidAlpha, SigmoidFlag, InverseFlag, getCLayer()->activeIndices, &getCLayer()->numActive);
   if( status == PV_SUCCESS ) status = updateActiveIndices();
   return status;
}

int SigmoidLayer::updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V,  int num_channels, pvdata_t * gSynHead, float Vth, float V0, float sigmoid_alpha, bool sigmoid_flag, bool inverse_flag, unsigned int * active_indices, unsigned int * num_active) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   updateV_SigmoidLayer(); // Does nothing as sourceLayer is responsible for updating V.
   setActivity_SigmoidLayer(num_neurons, A, V, nx, ny, nf, loc->nb, Vth, V0, sigmoid_alpha, sigmoid_flag, inverse_flag, dt);
   // resetGSynBuffers(); // Since sourceLayer updates V, this->GSyn is not used
   return PV_SUCCESS;
}

//int SigmoidLayer::updateV() {
//   return PV_SUCCESS;
//}

//int SigmoidLayer::resetGSynBuffers() {
//   return PV_SUCCESS;
//}


//int SigmoidLayer::setActivity() {
//
//   const int nx = getLayerLoc()->nx;
//   const int ny = getLayerLoc()->ny;
//   const int nf = getLayerLoc()->nf;
//   const int nb = getLayerLoc()->nb;
//   pvdata_t * activity = getCLayer()->activity->data;
//   pvdata_t * V = getV();
//   for( int k=0; k<getNumExtended(); k++ ) {
//      activity[k] = 0; // Would it be faster to only do the margins?
//   }
//   pvdata_t sig_scale = 1.0f;
//   if ( Vth > V0 ){
//      if(SigmoidFlag){
//      sig_scale = -0.5f * log(1.0f/SigmoidAlpha - 1.0f) / (Vth - V0);   // scale to get response alpha at Vrest
//      }
//      else{
//      sig_scale = 0.5/(Vth-V0);        // threshold in the middle
//      }
//   }
//   for( int k=0; k<getNumNeurons(); k++ ) {
//      int kex = kIndexExtended(k, nx, ny, nf, nb);
//
//      if(!SigmoidFlag) {
//         if (V[k] > 2*Vth-V0){    //  2x(Vth-V0) + V0
//            activity[kex] = 1.0f;
//         }
//         else if (V[k] < V0){
//            activity[kex] = 0.0f;
//         }
//         else{
//            activity[kex] = (V[k] - V0) * sig_scale;
//         }
//      }
//      else{
//         activity[kex] = 1.0f / (1.0f + exp(2.0f * (V[k] - Vth)*sig_scale));
//      }
//
//      if (InverseFlag) activity[kex] = 1.0f - activity[kex];
//   }
//
//   return PV_SUCCESS;
//
//}


} // end namespace PV

