/*
 * ReciprocalConn.cpp
 *
 *  Created on: Feb 16, 2012
 *      Author: pschultz
 */

#include "ReciprocalConn.hpp"

namespace PV {

ReciprocalConn::ReciprocalConn() {
   initialize_base();

}

ReciprocalConn::ReciprocalConn(const char * name, HyPerCol * hc,
         HyPerLayer * pre, HyPerLayer * post,
         ChannelType channel, const char * filename,
         InitWeights * weightInit) {
   initialize_base();
   initialize(name, hc, pre, post, channel, filename, weightInit);
}

int ReciprocalConn::initialize_base() {
   updateRulePre = NULL;
   updateRulePost = NULL;
   reciprocalWgts = NULL;
   slownessPre = NULL;
   slownessPost = NULL;
   return PV_SUCCESS;
}

int ReciprocalConn::initialize(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
      const char * filename, InitWeights * weightInit) {
   int status = PV_SUCCESS;
   status = KernelConn::initialize(name, hc, pre, post, channel, filename, weightInit);
   PVParams * params = hc->parameters();
   reciprocalFidelityCoeff = params->value(name, "reciprocalFidelityCoeff", 1);

   status = initParameterLayer("updateRulePre", &updateRulePre, pre) == PV_SUCCESS ? status : PV_FAILURE;
   status = initParameterLayer("updateRulePost", &updateRulePost, post) == PV_SUCCESS ? status : PV_FAILURE;

   slownessFlag = params->value(name, "slownessFlag", false)!=0;
   if( slownessFlag ) {
      status = initParameterLayer("slownessPre", &slownessPre, NULL) == PV_SUCCESS ? status : PV_FAILURE;
      status = initParameterLayer("slownessPost", &slownessPost, NULL) == PV_SUCCESS ? status : PV_FAILURE;
   }

   const char * reciprocalWgtsName = params->stringValue(name, "reciprocalWgts");
   if( reciprocalWgtsName == NULL || reciprocalWgtsName[0] == '0') {
      fprintf(stderr, "ReciprocalConn \"%s\": reciprocalWgts must be defined.\n", name);
      status = PV_FAILURE;
   }
   if( status != PV_SUCCESS ) abort();
   return status;
}

int ReciprocalConn::initParameterLayer(const char * parametername, HyPerLayer ** layerPtr, HyPerLayer * defaultlayer) {
   int status = PV_SUCCESS;
   PVParams * params = parent->parameters();
   const char * layerName = params->stringValue(name, parametername);
   if( layerName == NULL || layerName[0] == '0') {
      if( defaultlayer == NULL ) {
         fprintf(stderr, "ReciprocalConn \"%s\": parameter \"%s\" was not defined and no default was specified.\n", name, parametername);
         status = PV_FAILURE;
      }
      else {
         fprintf(stdout, "ReciprocalConn \"%s\": parameter \"%s\" set to \"s\"\n", name, defaultlayer->getName());
         *layerPtr = defaultlayer;
      }
   }
   else {
      *layerPtr = parent->getLayerFromName(layerName);
      if( *layerPtr == NULL ) {
         fprintf(stderr, "ReciprocalConn \"%s\" parameter \"%s\": value \"%s\" is not the name of a layer.\n", name, parametername, layerName);
         status = PV_FAILURE;
      }
   }
   return status;
}

int ReciprocalConn::updateState(float time, float dt) {
   // Need to set reciprocalWgts the first time updateState is called, so that each ReciprocalConn in a pair can define the other
   // If it was set in initialize, the second would not have been defined when the first was called.
   int status = PV_SUCCESS;
   if( reciprocalWgts == NULL) {
      setReciprocalWgts(reciprocalWgtsName);
   }
   return status;
}

int ReciprocalConn::setReciprocalWgts(const char * recipName) {
   int status;
   if( reciprocalWgts ) {
      fprintf(stderr, "ReciprocalConn \"%s\": setReciprocalWgts called with reciprocalWgts already set.\n", name);
      status = PV_FAILURE;
      abort();
   }
   else {
      HyPerConn * c = parent->getConnFromName(recipName);
      if( c == NULL) {
         status = PV_FAILURE;
         fprintf(stderr, "ReciprocalConn \"%s\": reciprocalWgts \"%s\" could not be found.\n", name, recipName);
      }
      else {
         status = PV_SUCCESS;
      }
   }
   if( status == PV_SUCCESS ) {
      reciprocalWgts = dynamic_cast<ReciprocalConn *>(parent->getConnFromName(recipName));
      if( reciprocalWgts == NULL ) {
         fprintf(stderr, "ReciprocalConn \"%s\": reciprocalWgts \"%s\" is not a ReciprocalConn.\n", name, recipName);
         status = PV_FAILURE;
      }
      else {
         status = PV_SUCCESS;
      }
   }
   return status;
}

int ReciprocalConn::update_dW(int axonID) {
   int nExt = preSynapticLayer()->getNumExtended();
   int numKernelIndices = getNumDataPatches();
   int delay = getDelay(axonID);
   const pvdata_t * preactbuf = updateRulePre->getLayerData(delay);
   const pvdata_t * postactbuf = updateRulePost->getLayerData(delay);
   const pvdata_t * slownessprebuf = getSlownessFlag() ? slownessPre->getLayerData(delay) : NULL;
   const pvdata_t * slownesspostbuf = getSlownessFlag() ? slownessPost->getLayerData(delay) : NULL;

   int sya = (post->getLayerLoc()->nf * (post->getLayerLoc()->nx + 2*post->getLayerLoc()->nb));
   // int syw = syp;
   for(int kExt=0; kExt<nExt;kExt++) {
      PVPatch * weights = getWeights(kExt,axonID);
      size_t offset = getAPostOffset(kExt, axonID);
      int ny = weights->ny;
      int nk = weights->nx * nfp;
      pvdata_t preact = preactbuf[kExt];
      const pvdata_t * postactRef = &postactbuf[offset];
      pvdata_t * dwdata = get_dwData(axonID, kExt);
      int lineoffsetw = 0;
      int lineoffseta = 0;
      for( int y=0; y<ny; y++ ) {
         for( int k=0; k<nk; k++ ) {
            dwdata[lineoffsetw + k] += updateRule_dW(preact, postactRef[lineoffseta+k]);
         }
         lineoffsetw += syp;
         lineoffseta += sya;
      }
      if( slownessFlag ) {
         preact = slownessprebuf[kExt];
         postactRef = &slownesspostbuf[offset];

         int lineoffsetw = 0;
         int lineoffseta = 0;
         for( int y=0; y<ny; y++ ) {
            for( int k=0; k<nk; k++ ) {
               dwdata[lineoffsetw + k] += updateRule_dW(preact, postactRef[lineoffseta+k]);
            }
            lineoffsetw += syp;
            lineoffseta += sya;
         }
      }
   }
   if( reciprocalFidelityCoeff ) {
      for( int k=0; k<numKernelIndices; k++) {
         const pvdata_t * wdata = get_wDataHead(axonID, k); // p->data;
         pvdata_t * dwdata = get_dwDataHead(axonID, k);
         // PVPatch * p = getWeights(k, axonID); // getKernelPatch(axonID, k);
         // short int nx = nxp; // p->nx;
         // short int ny = nyp; // p->ny;
         for( int n=0; n<nxp*nyp*nfp; n++ ) {
            int f = featureIndex(n,nxp,nyp,nfp);
            const pvdata_t * recipwdata = reciprocalWgts->get_wDataHead(axonID, f);
            // const pvdata_t * recipwdata = reciprocalWgts->getKernelPatch(axonID, f)->data;
            dwdata[n] += reciprocalFidelityCoeff*(wdata[n]-nfp/reciprocalWgts->fPatchSize()*recipwdata[k]);
         }
      }
   }

   // Divide by (numNeurons/numKernels)
   int divisor = pre->getNumNeurons()/numKernelIndices;
   assert( divisor*numKernelIndices == pre->getNumNeurons() );
   for( int kernelindex=0; kernelindex<numKernelIndices; kernelindex++ ) {
      // int patchIndex = kernelIndexToPatchIndex(kernelindex);
      int numpatchitems = nxp * nyp * nfp;
      // int numpatchitems = dKernelPatches[axonID][kernelindex]->nx * dKernelPatches[axonID][kernelindex]->ny * nfp;
      pvdata_t * dwpatchdata = get_dwDataHead(axonID,kernelindex);
      // pvdata_t * dwpatchdata = dKernelPatches[axonID][kernelindex]->data;
      for( int n=0; n<numpatchitems; n++ ) {
         dwpatchdata[n] /= divisor;
      }
   }

   lastUpdateTime = parent->simulationTime();
   return PV_SUCCESS;
}

int ReciprocalConn::normalizeWeights(PVPatch ** patches, pvdata_t * dataStart, int numPatches, int arborID) {
   assert(arborID == 0); // TODO how to handle arbors.  Do I need to sum over arbors or handle each arbor independently?
   int status = PV_SUCCESS;
   assert( numPatches == getNumDataPatches() );
   for( int f=0; f<nfp; f++ ) {
      pvdata_t sum = 0.0f;
      for( int k=0; k<numPatches; k++ ) {
         // int patchIndex = kernelIndexToPatchIndex(k);
         // PVPatch * w = getWeights(patchIndex,arborID);
         // int nx = (int) patches[k]->nx;
         // int ny = (int) patches[k]->ny;
         for( int m=0; m<nxp; m++ ) {
            for( int n=0; n<nyp; n++) {
               sum += dataStart[k*nxp*nyp*nfp+f];
               // sum += getKernelPatch(arborID, k)->data[f];
            }
         }
      }
      for( int k=0; k<numPatches; k++ ) {
         // int patchIndex = kernelIndexToPatchIndex(k);
         // PVPatch * w = getWeights(patchIndex,arborID);
         // int nx = (int) patches[k]->nx;
         // int ny = (int) patches[k]->ny;
         for( int m=0; m<nxp; m++ ) {
            for( int n=0; n<nyp; n++) {
               dataStart[k*nxp*nyp*nfp+f] /= sum;
               // getKernelPatch(arborID, k)->data[f] /= sum;
            }
         }
      }
   }
   return status;
}

ReciprocalConn::~ReciprocalConn() {
}

} /* namespace PV */
