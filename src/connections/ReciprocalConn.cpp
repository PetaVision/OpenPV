/*
 * ReciprocalConn.cpp
 *
 *  For two connections with transpose geometries (the pre of one has the
 *  same geometry as the post of another and vice versa).  They are not
 *  forced to be each other's transpose, but there is a penalty term
 *  on the L2 norm of the difference between it and its reciprocal's
 *  transpose, of the amount
 *  r/2 * || W/f_W - (W_recip)'/f_{W_recip} ||^2.
 *
 *  r is the parameter reciprocalFidelityCoeff
 *  W is the weights and W_recip is the weights of the reciprocal connection
 *  f_W and f_{W_recip} are the number of features of each connection.
 *
 *  The connection forces the normalization method to be that for each feature,
 *  the sum across kernels (number of features of reciprocal connection) is unity.
 *
 *  Dividing by f_W and f_{W_recip} therefore makes the sum of all matrix
 *  entries of each connection 1.
 *
 *  This is still under the assumption that nxp = nyp = 1.
 *
 *  There is also the ability to specify an additional penalty of the form
 *  dW/dt += slownessPost * slownessPre', by setting slownessFlag to true
 *  and specifying slownessPre and slownessPost layers.
 *
 *  The name comes from the motivation of including a slowness term on
 *  ReciprocalConns, but the slowness interpretation is not within ReciprocalConn.
 *  IncrementLayer was written so that setting A to an IncrementLayer, and
 *  having a layer B be the post of a clone of the ReciprocalConn, setting
 *  slownessPre to A and slownessPost to B would implement a slowness term.
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

   reciprocalWgtsName = params->stringValue(name, "reciprocalWgts");
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

int ReciprocalConn::initNormalize() {
   // ReciprocalConn only normalizes by setting for each feature the sum across all kernels to unity.
   // As of now there are no options for normalizing a ReciprocalConn
   return PV_SUCCESS;
}

int ReciprocalConn::updateState(float timef, float dt) {
   // Need to set reciprocalWgts the first time updateState is called, so that each ReciprocalConn in a pair can define the other
   // If it was set in initialize, the second would not have been defined when the first was called.
   if( reciprocalWgts == NULL) {
      setReciprocalWgts(reciprocalWgtsName);
   }
   int status = KernelConn::updateState(timef, dt);
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
         for( int n=0; n<nxp*nyp*nfp; n++ ) {
            int f = featureIndex(n,nxp,nyp,nfp);
            const pvdata_t * recipwdata = reciprocalWgts->get_wDataHead(axonID, f);
            dwdata[n] -= reciprocalFidelityCoeff/nfp*(wdata[n]/nfp-recipwdata[k]/reciprocalWgts->fPatchSize());
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
int ReciprocalConn::normalizeWeights(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborID) {
   assert(arborID == 0); // TODO how to handle arbors.  Do I need to sum over arbors or handle each arbor independently?
   int status = PV_SUCCESS;
   assert( numPatches == getNumDataPatches() );
   for( int f=0; f<nfp; f++ ) {
      pvdata_t sum = 0.0f;
      for( int k=0; k<numPatches; k++ ) {
         for( int m=0; m<nxp; m++ ) {
            for( int n=0; n<nyp; n++) {
               sum += dataStart[arborID][k*nxp*nyp*nfp+kIndex(m,n,f,nxp,nyp,nfp)];
            }
         }
      }
      for( int k=0; k<numPatches; k++ ) {
         for( int m=0; m<nxp; m++ ) {
            for( int n=0; n<nyp; n++) {
               dataStart[arborID][k*nxp*nyp*nfp+kIndex(m,n,f,nxp,nyp,nfp)] /= sum;
            }
         }
      }
   }
   return status;
}

ReciprocalConn::~ReciprocalConn() {
}

} /* namespace PV */
