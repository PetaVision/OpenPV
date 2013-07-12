/*
 * GapConn.cpp
 *
 *  Created on: Aug 2, 2011
 *      Author: garkenyon
 */

#include "GapConn.hpp"
#include "../layers/LIFGap.hpp"
#include "../normalizers/NormalizeBase.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace PV {

GapConn::GapConn()
{
   initialize_base();
}

GapConn::GapConn(const char * name, HyPerCol * hc,
      const char * pre_layer_name, const char * post_layer_name,
      const char * filename, InitWeights *weightInit) {
   // No channel argument to constructor because GapConn must always use CHANNEL_GAP
   initialize_base();
   GapConn::initialize(name, hc, pre_layer_name, post_layer_name, filename, weightInit);
}

GapConn::~GapConn()
{
}

int GapConn::initialize_base(){
   initNormalizeFlag = false;
   return PV_SUCCESS;
}

int GapConn::initialize(const char * name, HyPerCol * hc,
      const char * pre_layer_name, const char * post_layer_name,
      const char * filename, InitWeights *weightInit){
   return KernelConn::initialize(name, hc, pre_layer_name, post_layer_name, filename, weightInit);
}

void GapConn::readChannelCode(PVParams * params) {
   channel = CHANNEL_GAP;
}


int GapConn::initNormalize(){
   int status = KernelConn::initNormalize();
   if (normalizer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "GapConn::initNormalize error in connection \"%s\".  normalizeMethod cannot be \"none\".\n", name);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif // PV_USE_MPI
      exit(PV_FAILURE);
   }
   assert(normalizer);
   if (!normalizer->getNormalizeFromPostPerspectiveFlag()) {
      if (parent->columnId()==0) {
         fprintf(stderr, "GapConn::initNormalize error in connection \"%s\".  normalizeFromPostPerspective must be true for GapConns.\n", name);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif // PV_USE_MPI
      exit(PV_FAILURE);
   }
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
      assert(this->normalizer);
      gap_strength = normalizer->getStrength(); // normalizer->getStrength() / this->postSynapticLayer()->getNumNeurons() * this->preSynapticLayer()->getNumNeurons();
      //      fprintf(stdout,"This is connection %i, setting initNormalizeFlag to true and adding gap_strength %f \n",this->getConnectionId(),gap_strength);
      postLIFGap->addGapStrength(gap_strength);
   }
   return status;
}

} /* namespace PV */
