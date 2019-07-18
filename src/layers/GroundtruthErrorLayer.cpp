#include "GroundtruthErrorLayer.hpp"
#include "layers/updateStateFunctions.h"

namespace PV {
GroundtruthErrorLayer::GroundtruthErrorLayer() { initialize_base(); }

GroundtruthErrorLayer::GroundtruthErrorLayer(const char *name, HyPerCol *hc) {
  int status = initialize_base();
  if (status == PV_SUCCESS) {
    status = initialize(name, hc);
  }
  if (status != PV_SUCCESS) {
    Fatal().printf("Creating GroundtruthErrorLayer \"%s\" failed.\n", name);
  }
}

GroundtruthErrorLayer::~GroundtruthErrorLayer() {}

int GroundtruthErrorLayer::initialize_base() {
  thresholdCorrect = 1;
  thresholdIncorrect = 0;
  return PV_SUCCESS;
}

int GroundtruthErrorLayer::initialize(const char *name, HyPerCol *hc) {
  int status = HyPerLayer::initialize(name, hc);
  return status;
}

int GroundtruthErrorLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
  int status = HyPerLayer::ioParamsFillGroup(ioFlag);
  ioParam_thresholdCorrect(ioFlag);
  ioParam_thresholdIncorrect(ioFlag);
  return status;
}

void GroundtruthErrorLayer::ioParam_thresholdCorrect(enum ParamsIOFlag ioFlag) {
  parent->parameters()->ioParamValue(ioFlag, name, "thresholdCorrect",
                                     &thresholdCorrect, thresholdCorrect,
                                     true /*warnIfAbsent*/);
}

void GroundtruthErrorLayer::ioParam_thresholdIncorrect(
    enum ParamsIOFlag ioFlag) {
  parent->parameters()->ioParamValue(ioFlag, name, "thresholdIncorrect",
                                     &thresholdIncorrect, thresholdIncorrect,
                                     true /*warnIfAbsent*/);
}

int GroundtruthErrorLayer::updateState(double time, double dt) {
  const PVLayerLoc *loc = getLayerLoc();
  float *A = clayer->activity->data;
  float *V = getV();
  
  int nx = loc->nx;
  int ny = loc->ny;
  int nf = loc->nf;
  int numNeurons = nx * ny * nf;
  int nbatch            = loc->nbatch;

  int lt = loc->halo.lt;
  int rt = loc->halo.rt;
  int dn = loc->halo.dn;
  int up = loc->halo.up;

  MEM_GLOBAL float *GSynHead = GSyn == NULL ? NULL : GSyn[0];

  return groundtruthError(nbatch,numNeurons,GSynHead, A, nx,ny,nf,lt,rt,dn,up);
  
}
    
  // applyGSyn_HyPerLayer(nbatch, num_neurons, V, groundtruth);

  // Channel 0 is groundtruth, channel 1 is recon
  // for (int i = 0; i < nf; i++) {
  //   if (groundtruth[i] == thresholdCorrect) {
  //     if (recon[i] > thresholdCorrect) {
  //       A[i] = 0;
  //     } else {
  //       A[i] = V[i];
  //     }
  //   } else {
  //     if (recon[i] < thresholdIncorrect) {
  //       A[i] = 0;
  //     } else {
  //       A[i] = V[i];
  //     }
  //   }
  // }
  


}
