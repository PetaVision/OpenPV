/*
 * LCALayer.cpp
 *
 *  Created on: Sep 27, 2012
 *      Author: pschultz
 */

#include "LCALayer.hpp"

namespace PV {

LCALayer::LCALayer(const char * name, HyPerCol * hc, int num_channels) {
   initialize_base();
   initialize(name, hc, num_channels);
}

LCALayer::LCALayer() {
   initialize_base();
}

LCALayer::~LCALayer() {
   free(stimulus); stimulus = NULL;
}

int LCALayer::initialize_base() {
   stimulus = NULL;
   return PV_SUCCESS;
}

int LCALayer::initialize(const char * name, HyPerCol * hc, int num_channels) {
   int status = HyPerLayer::initialize(name, hc, num_channels);
   threshold = readThreshold();
   thresholdSoftness = readThresholdSoftness();
   timeConstantTau = readTimeConstantTau();

   // Moved to allocateDataStructures
   // stimulus = (pvdata_t *) calloc(getNumNeurons(), sizeof(pvdata_t));
   // if (stimulus == NULL) {
   //    fprintf(stderr, "LCALayer::initialize error allocating memory for stimulus: %s", strerror(errno));
   //    abort();
   // }

   return status;
}

int LCALayer::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();

   allocateBuffer(&stimulus, getNumNeurons(), "stimulus");

   return status;

}

int LCALayer::updateState(double timed, double dt) {
#define LCALAYER_FEEDBACK_LENGTH 3
#define LCALAYER_START_STEP 2
   int step = parent->getCurrentStep();
   if (step <= LCALAYER_START_STEP || step % LCALAYER_FEEDBACK_LENGTH != 0) return PV_SUCCESS;
   const pvdata_t * gSynExc = getChannel(CHANNEL_EXC);
   const pvdata_t * gSynInh = getChannel(CHANNEL_INH);
   pvdata_t * V = getV();
   pvdata_t * A = getActivity();
   const float dt_tau = LCALAYER_FEEDBACK_LENGTH*dt/timeConstantTau;
   const float threshdrop = thresholdSoftness * threshold;
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;
   const int nb = loc->nb;
   for (int k=0; k<getNumNeurons(); k++) {
      stimulus[k] = gSynExc[k] - gSynInh[k];
      int kex = kIndexExtended(k, nx, ny, nf, nb);
      pvdata_t Vk = V[k];
      // Corresponds to Eq 3.1 in Rozell et. al
      // A[k] term is due to gSynInh including n=m term
      Vk = Vk + dt_tau*(stimulus[k] - V[k] + A[kex]);
      A[kex] = Vk >= threshold ? Vk - threshdrop : 0.0;
      V[k] = Vk;
   }
   //resetGSynBuffers_HyPerLayer(getNumNeurons(), getNumChannels(), GSyn[0]);

   return PV_SUCCESS;
}

int LCALayer::checkpointWrite(const char * cpDir) {
   int status = HyPerLayer::checkpointWrite(cpDir);
   char filename[PV_PATH_MAX];
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_stimulus.pvp", cpDir, name);
   assert(chars_needed < PV_PATH_MAX);
   status = writeBufferFile(filename, getParent()->icCommunicator(), getParent()->simulationTime(), &stimulus, 1, /*extended*/false, getLayerLoc())
                == PV_SUCCESS ? status : PV_FAILURE;
   return status;
}

} /* namespace PV */


