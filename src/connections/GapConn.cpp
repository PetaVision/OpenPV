/*
 * GapConn.cpp
 *
 *  Created on: Aug 2, 2011
 *      Author: garkenyon
 */

#include "GapConn.hpp"
#include "../layers/LIFGap.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace PV {

GapConn::GapConn()
{
}

GapConn::GapConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, const char * filename, InitWeights *weightInit) {
   // No channel argument to constructor because GapConn must always use CHANNEL_GAP
   GapConn::initialize(name, hc, pre, post, filename, weightInit);
}

GapConn::~GapConn()
{
}

int GapConn::initialize(const char * name, HyPerCol * hc,
               HyPerLayer * pre, HyPerLayer * post,
               const char * filename,
               InitWeights *weightInit){
   initNormalizeFlag = false;
   return KernelConn::initialize(name, hc, pre, post, filename, weightInit);
}

ChannelType GapConn::readChannelCode(PVParams * params) {
   channel = CHANNEL_GAP;
   return channel;
}


int GapConn::initNormalize(){
   int status = KernelConn::initNormalize();
   HyPerLayer * postHyPerLayer = this->postSynapticLayer();
   LIFGap * postLIFGap = NULL;
   postLIFGap = dynamic_cast <LIFGap*> (postHyPerLayer);
   assert(postLIFGap != NULL);
   //   fprintf(stdout,"This is connection %i with flag %i \n",this->getConnectionId(),initNormalizeFlag);
   if (this->initNormalizeFlag == false){
      initNormalizeFlag = true;
      pvdata_t gap_strength;
      //TODO!!! terrible hack here: should compute sum of gap junctions connection strengths into each post synaptic cell
      // instead, we check that normalize is true as a stop gap
      assert(this->normalize_flag);
      gap_strength = this->normalize_strength / this->postSynapticLayer()->getNumNeurons() * this->preSynapticLayer()->getNumNeurons();
      //      fprintf(stdout,"This is connection %i, setting initNormalizeFlag to true and adding gap_strength %f \n",this->getConnectionId(),gap_strength);
      postLIFGap->addGapStrength(gap_strength);
   }
   return status;
}

} /* namespace PV */
