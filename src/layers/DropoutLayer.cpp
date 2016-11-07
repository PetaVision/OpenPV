#include "DropoutLayer.hpp"

namespace PV {

DropoutLayer::DropoutLayer(const char *name, HyPerCol *hc) {
   initialize(name, hc);
}

DropoutLayer::~DropoutLayer() { }

int DropoutLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_probability(ioFlag);
}

void DropoutLayer::ioParam_probability(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "probability", &mProbability, mProbability, true);
   if (mProbability >= 1.0f) {
      pvWarn() << getName() << ": probability was set to >= 100%. Changing to 99%.\n";
      mProbability = 0.99f;
   }
}


int DropoutLayer::updateState(double timestamp, double dt) {
   const PVLayerLoc *loc = getLayerLoc();

   float *A        = getCLayer()->activity->data;
   float *V        = getV();
   float *gSynHead = GSyn == NULL ? NULL : GSyn[0];

   int nx       = loc->nx;
   int ny       = loc->ny;
   int nf       = loc->nf;
   int nBatch   = loc->nbatch;
   int nNeurons = nx * ny * nf;

   if (getNumChannels() == 1) {
      applyGSyn_HyPerLayer1Channel(nBatch, nNeurons, V, gSynHead);
   }
   else {
      applyGSyn_HyPerLayer(nBatch, nNeurons, V, gSynHead);
   }

   // TODO: Change if more than 1% precision is needed
   // TODO: Don't hardcode RelU
   int thresh = (int)(mProbability * 100.0f);
   for (int i = 0; i < nNeurons * nBatch; ++i) {
      int b = i / nNeurons;
      int k = i % nNeurons;
      float *VBatch = V + b * nNeurons;
      if (VBatch[k] < 0 || rand() % 100 < thresh) {
         VBatch[k] = 0.0f;
      }
    }

   setActivity_HyPerLayer(
         nBatch,
         nNeurons,
         A,
         V,
         nx,
         ny,
         nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up);
}

}
