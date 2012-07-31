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
      HyPerLayer * post, ChannelType channel, const char * filename, InitWeights *weightInit) {
    initialize(name, hc, pre, post, channel, filename, weightInit);
}

GapConn::~GapConn()
{
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
      gap_strength = this->normalize_strength / this->postSynapticLayer()->getNumNeurons() * this->preSynapticLayer()->getNumNeurons();
//      fprintf(stdout,"This is connection %i, setting initNormalizeFlag to true and adding gap_strength %f \n",this->getConnectionId(),gap_strength);
      postLIFGap->addGapStrength(gap_strength);
   }
   return status;
}

} /* namespace PV */
