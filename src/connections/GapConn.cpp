/*
 * GapConn.cpp
 *
 *  Created on: Aug 2, 2011
 *      Author: garkenyon
 */

#include "GapConn.hpp"
#include "../layers/LIFGap.hpp"

namespace PV {

GapConn::GapConn()
{
}

GapConn::~GapConn()
{
}

int GapConn::initNormalize(){
   int status = HyPerConn::initNormalize();
   HyPerLayer * postHyPerLayer = this->postSynapticLayer();
   LIFGap * postLIFGap = NULL;
   postLIFGap = dynamic_cast <LIFGap*> (postHyPerLayer);
   assert(postLIFGap != NULL);
   if (this->initNormalizeFlag == false){
      initNormalizeFlag = true;
      pvdata_t gap_strength;
      gap_strength = this->normalize_strength * this->postSynapticLayer()->getNumNeurons() / this->preSynapticLayer()->getNumNeurons();
      postLIFGap->addGapStrength(gap_strength);
   }
   return status;
}

} /* namespace PV */
