/*
 * LCAConn.cpp
 *
 *  Created on: Oct 8, 2012
 *      Author: kpatel
 */

#include "LCAConn.hpp"
#include "../include/pv_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace PV {
  
  LCAConn::LCAConn(const char * name, HyPerCol * hc,
        const char * pre_layer_name, const char * post_layer_name,
		const char * filename, InitWeights *weightInit, const char * movieLayerName)
  {
    KernelConn::initialize_base();
    KernelConn::initialize(name, hc, pre_layer_name, post_layer_name, filename, weightInit);
    HyPerLayer * layer = parent->getLayerFromName(movieLayerName);
    layerOfInterest = dynamic_cast<Movie *>(layer);
    if (layerOfInterest==NULL) {
       fprintf(stderr, "LCAConn \"%s\" error: otherLayerName \"%s\" is not a Movie layer.\n", name, movieLayerName);
       exit(EXIT_FAILURE);
    }
  }
  
  int LCAConn::update_dW(int axonId)
  { // compute dW but don't add them to the weights yet.
    // That takes place in reduceKernels, so that the output is
    // independent of the number of processors.
    int nExt = preSynapticLayer()->getNumExtended();
    int numKernelIndices = getNumDataPatches();
    const pvdata_t * preactbuf = preSynapticLayer()->getLayerData(getDelay(axonId));
    const pvdata_t * postactbuf = postSynapticLayer()->getLayerData(getDelay(axonId));

    int sya = (post->getLayerLoc()->nf * (post->getLayerLoc()->nx + 2*post->getLayerLoc()->nb));
    
    for(int kExt=0; kExt<nExt;kExt++) {
      PVPatch * weights = getWeights(kExt,axonId);
      size_t offset = getAPostOffset(kExt, axonId);
      pvdata_t preact = preactbuf[kExt];
      int ny = weights->ny;
      int nk = weights->nx * nfp;
      const pvdata_t * postactRef = &postactbuf[offset];
      pvdata_t * dwdata = get_dwData(axonId, kExt);
      int lineoffsetw = 0;
      int lineoffseta = 0;
      for( int y=0; y<ny; y++ ) {
	for( int k=0; k<nk; k++ ) {
	  dwdata[lineoffsetw + k] += updateRule_dW(preact, postactRef[lineoffseta+k],lineoffseta+k);
	}
	lineoffsetw += syp;
	lineoffseta += sya;
      }
    }
    
    // Divide by (numNeurons/numKernels)
    int divisor = pre->getNumNeurons()/numKernelIndices;
    assert( divisor*numKernelIndices == pre->getNumNeurons() );
    for( int kernelindex=0; kernelindex<numKernelIndices; kernelindex++ ) {
      int numpatchitems = nxp*nyp*nfp;
      pvdata_t * dwpatchdata = get_dwDataHead(axonId,kernelindex);
      for( int n=0; n<numpatchitems; n++ ) {
	dwpatchdata[n] /= divisor;
      }
    }

    lastUpdateTime = parent->simulationTime();

    return PV_SUCCESS;    
  }

  pvdata_t LCAConn::updateRule_dW(pvdata_t preact, pvdata_t postact) {
     return KernelConn::updateRule_dW(preact, postact);
  }

  pvdata_t LCAConn::updateRule_dW(pvdata_t preact, pvdata_t postact, int offset)
  {
    pvdata_t * image = layerOfInterest->getImageBuffer();
    const pvdata_t * recon = postSynapticLayer()->getLayerData();
    float beta = 0.001;    
    return beta*(image[offset] - recon[offset])*preact;
  }

}


