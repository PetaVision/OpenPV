/*
 * GenerativeConn.cpp
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#include "GenerativeConn.hpp"

namespace PV {

GenerativeConn::GenerativeConn() {
    initialize_base();
}  // end of GenerativeConn::GenerativeConn()

GenerativeConn::GenerativeConn(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, ChannelType channel) {
       initialize_base();
       initialize(name, hc, pre, post, channel);
}  // end of GenerativeConn::GenerativeConn(const char *, HyPerCol *,
   //   HyPerLayer *, HyPerLayer *, int)

GenerativeConn::GenerativeConn(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
        const char * filename) {
       initialize_base();
       initialize(name, hc, pre, post, channel, filename);
}  // end of GenerativeConn::GenerativeConn(const char *, HyPerCol *,
   //   HyPerLayer *, HyPerLayer *, int, const char *)

int GenerativeConn::initialize_base() {
   plasticityFlag = true; // Default value; override in params
   weightUpdatePeriod = 1;   // Default value; override in params

   relaxation = 1.0;
   nonnegConstraintFlag = false;
   normalizeMethod = 0;
   return PV_SUCCESS;
   // Base class constructor calls base class initialize_base
   // so derived class initialize_base doesn't need to.
}

int GenerativeConn::initialize(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post, ChannelType channel) {
   return initialize(name, hc, pre, post, channel, NULL);
}

int GenerativeConn::initialize(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
        const char * filename) {
    PVParams * params = hc->parameters();
    relaxation = params->value(name, "relaxation", 1.0f);
    nonnegConstraintFlag = (bool) params->value(name, "nonnegConstraintFlag", 0.f); // default is not to constrain nonnegative.
    KernelConn::initialize(name, hc, pre, post, channel, filename);
    return PV_SUCCESS;
}

int GenerativeConn::updateWeights(int axonID) {
    int nPre = preSynapticLayer()->getNumNeurons();
    int nx = preSynapticLayer()->getLayerLoc()->nx;
    int ny = preSynapticLayer()->getLayerLoc()->ny;
    int nf = preSynapticLayer()->getLayerLoc()->nf;
    int pad = preSynapticLayer()->getLayerLoc()->nb;
    for(int kPre=0; kPre<nPre;kPre++) {
        int kExt = kIndexExtended(kPre, nx, ny, nf, pad);

        PVAxonalArbor * arbor = axonalArbor(kPre, 0);
        size_t offset = arbor->offset;
        pvdata_t preact = preSynapticLayer()->getCLayer()->activity->data[kExt];
        pvdata_t a = relaxation*preact;
        int nyp = arbor->weights->ny;
        int nk = arbor->weights->nx * arbor->weights->nf;
        pvdata_t * postactRef = &(postSynapticLayer()->getCLayer()->activity->data[offset]);
        int sya = arbor->data->sy;
        pvdata_t * wtpatch = arbor->weights->data;
        int syw = arbor->weights->sy;
        for( int y=0; y<nyp; y++ ) {
            int lineoffsetw = 0;
            int lineoffseta = 0;
            for( int k=0; k<nk; k++ ) {
               float w = wtpatch[lineoffsetw + k] + a*postactRef[lineoffseta + k];
               if( nonnegConstraintFlag && w < 0) w = 0;
               wtpatch[lineoffsetw + k] = w;
            }
            lineoffsetw += syw;
            lineoffseta += sya;
        }
    }
    normalizeWeights( kernelPatches, numDataPatches(0) );
    lastUpdateTime = parent->simulationTime();

    return PV_SUCCESS;
}  // end of GenerativeConn::updateWeights(int);

int GenerativeConn::initNormalize() {
   PVParams * params = parent->parameters();
   normalizeMethod = (int) params->value(name, "normalizeMethod", 0.f); // default is no constraint
   switch( normalizeMethod ) {
   case 0:
      break;
   case 1:
      KernelConn::initNormalize();
      break;
   case 2: // fallthrough is intentional
   case 3:
      normalizeConstant = params->value(name, "normalizeConstant", 1.0f);
      break;
   default:
      fprintf(stderr,"Connection \"%s\": Unrecognized normalizeMethod %d.  Using normalizeMethod = 1 (KernelConn's normalizeWeights).\n", this->getName(), normalizeMethod);
      KernelConn::initNormalize();
      break;
   }
   return PV_SUCCESS;
}

PVPatch ** GenerativeConn::normalizeWeights(PVPatch ** patches, int numPatches) {
   int neuronsperpatch;
   switch( normalizeMethod ) {
   case 0:
      break;
   case 1:
      patches = KernelConn::normalizeWeights(patches, numPatches);
      break;
   case 2:
      neuronsperpatch = (patches[0]->nx)*(patches[0]->ny)*(patches[0]->nf);
      for( int n=0; n<neuronsperpatch; n++ ) {
         pvdata_t s = 0;
         for( int k=0; k<numPatches; k++ ) {
            pvdata_t d = patches[k]->data[n];
            s += d*d;
         }
         for( int k=0; k<numPatches; k++ ) {
            patches[k]->data[n] *= normalizeConstant/sqrt(s);
         }
      }
      break;
   case 3:
      neuronsperpatch = (patches[0]->nx)*(patches[0]->ny)*(patches[0]->nf);
      for( int k=0; k<numPatches; k++ ) {
         PVPatch * curpatch = patches[k];
         pvdata_t s = 0;
         for( int n=0; n<neuronsperpatch; n++ ) {
            pvdata_t d = curpatch->data[n];
            s += d*d;
         }
         for( int n=0; n<neuronsperpatch; n++ ) {
            curpatch->data[n] *= normalizeConstant/sqrt(s);
         }
      }
      break;
   default:
      assert(false); // This possibility was eliminated in initNormalize().
      break;
   }
   return patches;
}  // end of GenerativeConn::normalizeWeights

}  // end of namespace PV block
