
/*
 * SoftMaxBackprop.cpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist 
 */

#include "SoftMaxBackprop.hpp"

namespace PV {

SoftMaxBackprop::SoftMaxBackprop()
{
   initialize_base();
}

SoftMaxBackprop::SoftMaxBackprop(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

SoftMaxBackprop::~SoftMaxBackprop()
{
   if(forwardLayername) free(forwardLayername);
   forwardLayer = NULL;
   clayer->V = NULL;
}

int SoftMaxBackprop::initialize_base()
{
   forwardLayername = NULL;
   return PV_SUCCESS;
}

int SoftMaxBackprop::initialize(const char * name, HyPerCol * hc)
{
   int status = ANNLayer::initialize(name, hc);
   return status;
}

int SoftMaxBackprop::communicateInitInfo(){
   forwardLayer = parent->getLayerFromName(forwardLayername);
   if (forwardLayer ==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: ForwardLayername \"%s\" is not a layer in the HyPerCol.\n",
                 parent->parameters()->groupKeywordFromName(name), name, forwardLayername);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
   if (!forwardLayer->getInitInfoCommunicatedFlag()) {
      return PV_POSTPONE;
   }

   int status = ANNLayer::communicateInitInfo();

   const PVLayerLoc * srcLoc = forwardLayer->getLayerLoc();
   const PVLayerLoc * loc = getLayerLoc();
   assert(srcLoc != NULL && loc != NULL);
   if (srcLoc->nxGlobal != loc->nxGlobal || srcLoc->nyGlobal != loc->nyGlobal || srcLoc->nf != loc->nf) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: ForwardLayerName \"%s\" does not have the same dimensions.\n",
                 parent->parameters()->groupKeywordFromName(name), name, forwardLayername);
         fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                 srcLoc->nxGlobal, srcLoc->nyGlobal, srcLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
   assert(srcLoc->nx==loc->nx && srcLoc->ny==loc->ny);
   return status;
}

int SoftMaxBackprop::allocateDataStructures() {
   assert(forwardLayer);
   int status = PV_SUCCESS;
   // Make sure forwardLayer has allocated its V buffer before copying its address to clone's V buffer
   if (forwardLayer->getDataStructuresAllocatedFlag()) {
      status = HyPerLayer::allocateDataStructures();
   }
   else {
      status = PV_POSTPONE;
   }
   return status;
}

int SoftMaxBackprop::allocateV() {
   assert(forwardLayer && forwardLayer->getCLayer());
   //Set own V to forward layer's V
   clayer->V = forwardLayer->getV();
   if (getV()==NULL) {
      fprintf(stderr, "%s \"%s\": forwardLayer \"%s\" has a null V buffer in rank %d process.\n",
              parent->parameters()->groupKeywordFromName(name), name, forwardLayername, parent->columnId());
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

int SoftMaxBackprop::initializeV() {
   return PV_SUCCESS;
}

int SoftMaxBackprop::readVFromCheckpoint(const char * cpDir, double * timeptr) {
   // If we just inherit HyPerLayer::readVFromCheckpoint, we checkpoint V since it is non-null.  This is redundant since V is a clone.
   return PV_SUCCESS;
}

int SoftMaxBackprop::checkpointWrite(const char * cpDir) {
   pvdata_t * V = clayer->V;
   clayer->V = NULL;
   int status = HyPerLayer::checkpointWrite(cpDir);
   clayer->V = V;
   return status;
}

int SoftMaxBackprop::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_ForwardLayername(ioFlag);
   return status;
}

void SoftMaxBackprop::ioParam_ForwardLayername(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "ForwardLayername", &forwardLayername);
}

int SoftMaxBackprop::updateState(double time, double dt)
{
   //update_timer->start();
   //assert(getNumChannels()>= 3);

   const PVLayerLoc * loc = getLayerLoc();

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   //Reset pointer of gSynHead to point to the inhib channel
   pvdata_t * GSynExt = getChannel(CHANNEL_EXC);
   pvdata_t * GSynInh = getChannel(CHANNEL_INH);


   pvdata_t * A = getCLayer()->activity->data;
   pvdata_t * V = getV();
   float sumexp = 0;

   //Here, channel 0 is gt, 1 is estimate, and current V is what got softmaxed
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction( + : sumexp)
#endif
   for(int ni = 0; ni < num_neurons; ni++){
      sumexp += exp(V[ni]);
   }

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for(int ni = 0; ni < num_neurons; ni++){
      int next = kIndexExtended(ni, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      //Update activity
      //f'(V)*(error)
      //error = gt - finalLayer iff error is last error
      float errProp, gradient;
      //exct is expected, inh is actual
      //0 is DCR
      //expected - actual
      errProp = GSynExt[ni] - GSynInh[ni];
      float constant = sumexp - V[ni];
      gradient = (exp(V[ni]) * constant)/sumexp;
      A[next] = errProp * gradient;
   }
   //update_timer->stop();
   return PV_SUCCESS;
}

} /* namespace PV */
