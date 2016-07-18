/*
 * CPTestInputLayer.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#include "CPTestInputLayer.hpp"
#include "CPTest_updateStateFunctions.h"

void CPTestInputLayer_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      float * V,
      const float Vth,
      const float AMin,
      const float AMax,
      float * GSynHead,
      /*    float * GSynExc,
    float * GSynInh,*/
      float * activity);


namespace PV {

CPTestInputLayer::CPTestInputLayer(const char * name, HyPerCol * hc) {
   initialize(name, hc);
}

CPTestInputLayer::~CPTestInputLayer() {
}

int CPTestInputLayer::initialize(const char * name, HyPerCol * hc) {
   ANNLayer::initialize(name, hc);
   return PV_SUCCESS;
}

int CPTestInputLayer::allocateDataStructures() {
   int status = ANNLayer::allocateDataStructures();
   if (status!=PV_SUCCESS) return status;

   status = initializeV();
   if (status != PV_SUCCESS) {
      pvErrorNoExit().printf("CPTestInputLayer \"%s\" in rank %d process: initializeV failed.\n", name, parent->columnId());
   }
   return status;
}

int CPTestInputLayer::initializeV() {
   assert(parent->parameters()->value(name, "restart", 0.0f, false)==0.0f); // initializeV should only be called if restart is false
   const PVLayerLoc * loc = getLayerLoc();
   for (int b = 0; b < parent->getNBatch(); b++){
      pvdata_t * VBatch = getV() + b * getNumNeurons();
      for (int k = 0; k < getNumNeurons(); k++){
         int kx = kxPos(k,loc->nx,loc->nx,loc->nf);
         int ky = kyPos(k,loc->nx,loc->ny,loc->nf);
         int kf = featureIndex(k,loc->nx,loc->ny,loc->nf);
         int kGlobal = kIndex(loc->kx0+kx,loc->ky0+ky,kf,loc->nxGlobal,loc->nyGlobal,loc->nf);
         VBatch[k] = (pvdata_t) kGlobal;
      }
   }
   return PV_SUCCESS;
}



int CPTestInputLayer::updateState(double timed, double dt) {
   update_timer->start();
   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->loc.nf;
   const PVHalo * halo = &clayer->loc.halo;
   const int numNeurons = getNumNeurons();
   const int nbatch = clayer->loc.nbatch;

   pvdata_t * GSynHead   = GSyn[0];
   pvdata_t * V = getV();
   pvdata_t * activity = clayer->activity->data;

   CPTestInputLayer_update_state(nbatch, numNeurons, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up, V, VThresh, AMin, AMax, GSynHead, activity);

   update_timer->stop();
   return PV_SUCCESS;
}

BaseObject * createCPTestInputLayer(char const * name, HyPerCol * hc) {
   return hc ? new CPTestInputLayer(name, hc) : NULL;
}

}  // end of namespace PV block

//Kernel
void CPTestInputLayer_update_state(
    const int nbatch,
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    float * V,
    const float Vth,
    const float AMin,
    const float AMax,
    float * GSynHead,
    float * activity)
{
   updateV_CPTestInputLayer(nbatch, numNeurons, V);
   setActivity_HyPerLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
   resetGSynBuffers_HyPerLayer(nbatch, numNeurons, 2, GSynHead);
}

