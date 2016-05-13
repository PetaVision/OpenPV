/*
 * LIFGap.cpp
 *
 *  Created on: Jul 29, 2011
 *      Author: garkenyon
 */

#include "LIFGap.hpp"
#include "../include/pv_common.h"
#include "../include/default_params.h"
#include "../io/fileio.hpp"
#include "../connections/HyPerConn.hpp"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#ifdef __cplusplus
extern "C" {
#endif

void LIFGap_update_state_original(
    const int nbatch,
    const int numNeurons,
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    LIF_params * params,
    taus_uint4 * rnd,

    float * V,
    float * Vth,
    float * G_E,
    float * G_I,
    float * G_IB,
    float * GSynHead,
    float * activity,

    const pvgsyndata_t * gapStrength
);

void LIFGap_update_state_beginning(
    const int nbatch,
    const int numNeurons,
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    LIF_params * params,
    taus_uint4 * rnd,

    float * V,
    float * Vth,
    float * G_E,
    float * G_I,
    float * G_IB,
    float * GSynHead,
    float * activity,

    const pvgsyndata_t * gapStrength
);

void LIFGap_update_state_arma(
    const int nbatch,
    const int numNeurons,
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    LIF_params * params,
    taus_uint4 * rnd,

    float * V,
    float * Vth,
    float * G_E,
    float * G_I,
    float * G_IB,
    float * GSynHead,
    float * activity,

    const pvgsyndata_t * gapStrength
);


#ifdef __cplusplus
}
#endif


namespace PV {

LIFGap::LIFGap() {
   initialize_base();
}

LIFGap::LIFGap(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc, "LIFGap_update_state");
//#ifdef PV_USE_OPENCL
//   if(gpuAccelerateFlag)
//      initializeGPU();
//#endif
}

LIFGap::~LIFGap()
{
   free(gapStrength);
//#ifdef PV_USE_OPENCL
//   if(gpuAccelerateFlag) {
//      delete clG_Gap;
//      delete clGSynGap;
//   }
//#endif

}

int LIFGap::initialize_base() {
   numChannels = 4;
   gapStrength = NULL;
   gapStrengthInitialized = false;
//#ifdef PV_USE_OPENCL
//   clG_Gap = NULL;
//   clGSynGap = NULL;
//#endif

   return PV_SUCCESS;
}

// Initialize this class
/*
 *
 */
int LIFGap::initialize(const char * name, HyPerCol * hc, const char * kernel_name) {
   int status = LIF::initialize(name, hc, kernel_name);

//#ifdef PV_USE_OPENCL
//   numEvents=NUM_LIFGAP_EVENTS;
//#endif

   return status;
}


//#ifdef PV_USE_OPENCL
///**
// * Initialize OpenCL buffers.  This must be called after PVLayer data have
// * been allocated.
// */
//int LIFGap::initializeThreadBuffers(const char * kernel_name)
//{
//   int status = CL_SUCCESS;
//
//   status = LIF::initializeThreadBuffers(kernel_name);
//
//   const size_t size    = getNumNeurons()  * sizeof(pvdata_t);
//   CLDevice * device = parent->getCLDevice();
//
//   // these buffers are shared between host and device
//   //
//
//   clGSynGap = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, getChannel(CHANNEL_GAP));
//
//   return status;
//}
//
//int LIFGap::initializeThreadKernels(const char * kernel_name)
//
//{
//   int status = CL_SUCCESS;
//
//   status = LIF::initializeThreadKernels(kernel_name);
//
//   int argid = getNumKernelArgs();
//
//   status |= krUpdate->setKernelArg(argid++, sumGap);
//   status |= krUpdate->setKernelArg(argid++, clG_Gap);
//   //status |= krUpdate->setKernelArg(argid++, clGSynGap);
//
//   numKernelArgs = argid;
//
//   return status;
//}
//#endif

int LIFGap::allocateConductances(int num_channels) {
   // this->sumGap = 0.0f;
   int status = LIF::allocateConductances(num_channels-1); // CHANNEL_GAP doesn't have a conductance per se.
   gapStrength = (pvgsyndata_t *) calloc((size_t) getNumNeuronsAllBatches(), sizeof(*gapStrength));
   if(gapStrength == NULL) {
      fprintf(stderr, "%s layer \"%s\": rank %d process unable to allocate memory for gapStrength: %s\n",
              getKeyword(), getName(), parent->columnId(), strerror(errno));
      exit(EXIT_FAILURE);
   }
   return status;
}

int LIFGap::calcGapStrength() {
   bool needsNewCalc = !gapStrengthInitialized;
   if (!needsNewCalc) {
      for (int c=0; c<parent->numberOfConnections(); c++) {
         HyPerConn * conn = dynamic_cast<HyPerConn *>(parent->getConnection(c));
         if (conn->postSynapticLayer() != this || conn->getChannel() != CHANNEL_GAP) { continue; }
         if (lastUpdateTime < conn->getLastUpdateTime()) {
            needsNewCalc = true;
            break;
         }
      }
   }
   if (!needsNewCalc) { return PV_SUCCESS; }

   for (int k=0; k<getNumNeuronsAllBatches(); k++) {
      gapStrength[k] = (pvgsyndata_t) 0;
   }
   for (int c=0; c<parent->numberOfConnections(); c++) {
      HyPerConn * conn = dynamic_cast<HyPerConn *>(parent->getConnection(c));
      if (conn->postSynapticLayer() != this || conn->getChannel() != CHANNEL_GAP) { continue; }
      if (conn->getPlasticityFlag() && parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" warning: connection \"%s\" on CHANNEL_GAP has plasticity flag set to true\n", getKeyword(), getName(), conn->getName());
      }
      HyPerLayer * pre = conn->preSynapticLayer();
      const int sy = conn->getPostNonextStrides()->sy;
      const int syw = conn->yPatchStride();
      for (int arbor=0; arbor<conn->numberOfAxonalArborLists(); arbor++) {
         for (int k=0; k<pre->getNumExtendedAllBatches(); k++) {
            conn->deliverOnePreNeuronActivity(k, arbor, (pvadata_t) 1.0, gapStrength, NULL);
         }
      }
   }
   gapStrengthInitialized = true;
   return PV_SUCCESS;
}

int LIFGap::checkpointWrite(const char * cpDir) {
   int status = LIF::checkpointWrite(cpDir);

   // checkpoint gapStrength buffer
   InterColComm * icComm = parent->icCommunicator();
   double timed = (double) parent->simulationTime();
   int filenamesize = strlen(cpDir)+(size_t) 1+strlen(name)+strlen("_gapStrength.pvp")+(size_t) 1;
   // The +1's are for the slash between cpDir and name, and for the null terminator
   char * filename = (char *) malloc( filenamesize*sizeof(char) );
   assert(filename != NULL);
   int chars_needed;

   chars_needed = snprintf(filename, filenamesize, "%s/%s_gapStrength.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile(filename, icComm, timed, &gapStrength, 1, /*extended*/false, getLayerLoc());
   free(filename);
   return status;
}

int LIFGap::readStateFromCheckpoint(const char * cpDir, double * timeptr) {
   int status = LIF::readStateFromCheckpoint(cpDir, timeptr);
   status = readGapStrengthFromCheckpoint(cpDir, timeptr);
   return status;
}

int LIFGap::readGapStrengthFromCheckpoint(const char * cpDir, double * timeptr) {
   char * filename = parent->pathInCheckpoint(cpDir, getName(), "_gapStrength.pvp");
   int status = readBufferFile(filename, parent->icCommunicator(), timeptr, &gapStrength, 1, /*extended*/false, getLayerLoc());
   assert(status==PV_SUCCESS);
   free(filename);
   gapStrengthInitialized = true;
   return status;
}

int LIFGap::updateStateOpenCL(double time, double dt)
{
   int status = 0;

//#ifdef PV_USE_OPENCL
//   status = LIF::updateStateOpenCL(time, dt);
//
//#if PV_CL_COPY_BUFFERS
//   status |= clGSynGap->copyFromDevice(1, &evUpdate, &evList[getEVGSynGap()]);
//#endif
//
//   //do we need to copy gap back and forth?
//   //status |= clG_Gap->copyFromDevice(1, &evUpdate, &evList[EV_LIF_GSYN_GAP]);
////   status |= getChannelCLBuffer(CHANNEL_GAP)->copyFromDevice(1, &evUpdate, &evList[getEVGSynGap()]);
////   numWait += 1;
//#endif

   return status;
}

int LIFGap::updateState(double time, double dt)
{
   int status = PV_SUCCESS;
   //update_timer->start();

   status = calcGapStrength();

//#ifdef PV_USE_OPENCL
//   if((gpuAccelerateFlag)&&(true)) {
//      updateStateOpenCL(time, dt);
//   }
//   else {
//#endif

   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->loc.nf;
   const PVHalo * halo = &clayer->loc.halo;
   const int nbatch = clayer->loc.nbatch;

   pvdata_t * GSynHead   = GSyn[0];
   pvdata_t * activity = clayer->activity->data;

   switch (method) {
   case 'a':
      LIFGap_update_state_arma(nbatch, getNumNeurons(), time, dt, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up, &lParams, randState->getRNG(0), clayer->V, Vth, G_E,
            G_I, G_IB, GSynHead, activity, gapStrength);
   break;
   case 'b':
      LIFGap_update_state_beginning(nbatch, getNumNeurons(), time, dt, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up, &lParams, randState->getRNG(0), clayer->V, Vth, G_E,
            G_I, G_IB, GSynHead, activity, gapStrength);
   break;
   case 'o':
      LIFGap_update_state_original(nbatch, getNumNeurons(), time, dt, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up, &lParams, randState->getRNG(0), clayer->V, Vth, G_E,
            G_I, G_IB, GSynHead, activity, gapStrength);
      break;
   default:
      assert(0);
      break;
   }
//#ifdef PV_USE_OPENCL
//   }
//#endif
   //update_timer->stop();
   return status;
}

BaseObject * createLIFGap(char const * name, HyPerCol * hc) {
   return hc ? new LIFGap(name, hc) : NULL;
}

} // namespace PV

///////////////////////////////////////////////////////
//
// implementation of LIF kernels
//

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/LIFGap_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/LIFGap_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif
