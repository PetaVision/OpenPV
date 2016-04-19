/*
 * MLPSigmoidLayer.cpp
 *
 */

#include "MLPSigmoidLayer.hpp"
#include <stdio.h>

#include <include/default_params.h>

// MLPSigmoidLayer can be used to implement Sigmoid junctions
namespace PVMLearning {
MLPSigmoidLayer::MLPSigmoidLayer() {
   initialize_base();
}

MLPSigmoidLayer::MLPSigmoidLayer(const char * name, PV::HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

MLPSigmoidLayer::~MLPSigmoidLayer()
{
   // Handled by CloneVLayer destructor
   // clayer->V = NULL;
   // free(sourceLayerName);
}

int MLPSigmoidLayer::initialize_base() {
   // Handled by CloneVLayer
   // sourceLayerName = NULL;
   // sourceLayer = NULL;
   linAlpha = 0;
   symSigmoid = true;
   return PV_SUCCESS;
}

int MLPSigmoidLayer::initialize(const char * name, PV::HyPerCol * hc) {
   int status_init = CloneVLayer::initialize(name, hc);

   return status_init;
}

int MLPSigmoidLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = CloneVLayer::ioParamsFillGroup(ioFlag);
   ioParam_SymSigmoid(ioFlag);
   if(symSigmoid){
      ioParam_LinAlpha(ioFlag);
   }
   else{
      ioParam_Vrest(ioFlag);
      ioParam_VthRest(ioFlag);
      ioParam_InverseFlag(ioFlag);
      ioParam_SigmoidFlag(ioFlag);
      ioParam_SigmoidAlpha(ioFlag);
   }
   return status;
}

void MLPSigmoidLayer::ioParam_LinAlpha(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "linAlpha", &linAlpha, linAlpha);
}

void MLPSigmoidLayer::ioParam_SymSigmoid(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "symSigmoid", &symSigmoid, symSigmoid);
}

void MLPSigmoidLayer::ioParam_Vrest(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "Vrest", &V0, (float) V_REST);
}
void MLPSigmoidLayer::ioParam_VthRest(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "VthRest", &Vth, (float) VTH_REST);
}
void MLPSigmoidLayer::ioParam_InverseFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "InverseFlag", &InverseFlag, (bool) INVERSEFLAG);
}
void MLPSigmoidLayer::ioParam_SigmoidFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "SigmoidFlag", &SigmoidFlag, (bool) SIGMOIDFLAG);
}
void MLPSigmoidLayer::ioParam_SigmoidAlpha(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "SigmoidAlpha", &SigmoidAlpha, (float) SIGMOIDALPHA);
}

int MLPSigmoidLayer::communicateInitInfo() {
   int status = CloneVLayer::communicateInitInfo();
   assert(originalLayer);
   MLPForwardLayer * forwardLayer = dynamic_cast<MLPForwardLayer*>(originalLayer);
   if(!forwardLayer){
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: Original layer \"%s\" need to be a MLPForwardLayer.\n",
                 getKeyword(), name, originalLayerName);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
   //Set pointer to forward layer's dropout buffer
   dropout = forwardLayer->getDropout();
   return status;
}

int MLPSigmoidLayer::allocateDataStructures() {
   int status = CloneVLayer::allocateDataStructures();
   // Should have been initialized with zero channels, so GSyn should be NULL and freeChannels() call should be unnecessary
   assert(GSyn==NULL);
   return status;
}


int MLPSigmoidLayer::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtended);
   return 0;
}

int MLPSigmoidLayer::updateState(double timef, double dt) {
   int status;
   status = updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), 0, NULL, Vth, V0, SigmoidAlpha, SigmoidFlag, InverseFlag, linAlpha, dropout);
   return status;
}

int MLPSigmoidLayer::updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, float Vth, float V0, float sigmoid_alpha, bool sigmoid_flag, bool inverse_flag, float linear_alpha, bool* dropout_buf) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   updateV_SigmoidLayer(); // Does nothing as sourceLayer is responsible for updating V.
   if(symSigmoid){
      setActivity_MLPSigmoidLayer(loc->nbatch, num_neurons, A, V, linear_alpha, dropout_buf, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, dt);
   }
   else{
      //TODO implement dropout here
      setActivity_SigmoidLayer(loc->nbatch, num_neurons, A, V, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, Vth, V0, sigmoid_alpha, sigmoid_flag, inverse_flag, dt);
   }
   // resetGSynBuffers(); // Since sourceLayer updates V, this->GSyn is not used
   return PV_SUCCESS;
}

PV::BaseObject * createMLPSigmoidLayer(char const * name, PV::HyPerCol * hc) { 
   return hc ? new MLPSigmoidLayer(name, hc) : NULL;
}


} // end namespace PVMLearning
