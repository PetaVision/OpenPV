/*
 * GapConn.cpp
 *
 *  Created on: Aug 2, 2011
 *      Author: garkenyon
 */

#include "GapConn.hpp"
#include "../layers/LIFGap.hpp"
#include "../normalizers/NormalizeGap.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace PV {

GapConn::GapConn()
{
   initialize_base();
}

GapConn::GapConn(const char * name, HyPerCol * hc) {
   initialize_base();
   GapConn::initialize(name, hc);
}

GapConn::~GapConn()
{
}

int GapConn::initialize_base(){
   initNormalizeFlag = false;
   return PV_SUCCESS;
}

int GapConn::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerConn::initialize(name, hc);
   assert(dynamic_cast<NormalizeGap *>(normalizer));
   assert(normalizer->getNormalizeFromPostPerspectiveFlag());
   return status;
}

void GapConn::ioParam_channelCode(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      channel = CHANNEL_GAP;
      parent->parameters()->handleUnnecessaryParameter(name, "channelCode", (int) CHANNEL_GAP);
   }
}

// TODO: make sure code works in non-shared weight case
void GapConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   sharedWeights = true;
   if (ioFlag == PARAMS_IO_READ) {
      fileType = PVP_KERNEL_FILE_TYPE;
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", true/*correctValue*/);
   }
}

void GapConn::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "normalizeMethod", "normalizeSum", false);
      normalizer = new NormalizeGap(this);
   }
}

int GapConn::allocateDataStructures() {
   // We have to wait until postsynaptic LIFGap has called its allocateDataStructures before we call its addGapStrength method,
   // because LIFGap sets sumGap to zero in allocateDataStructures.  It may be possible to have LIFGap set sumGap to zero in
   // initialize_base, and move this code to GapConn::communicateInitInfo where it really belongs.
   if (!post->getDataStructuresAllocatedFlag()) {
      if (parent->columnId()==0) {
         const char * connectiontype = parent->parameters()->groupKeywordFromName(name);
         printf("%s \"%s\" must wait until post-synaptic layer \"%s\" has finished its allocateDataStructures stage.\n", connectiontype, name, post->getName());
      }
      return PV_POSTPONE;
   }
   HyPerLayer * postHyPerLayer = this->postSynapticLayer();
   LIFGap * postLIFGap = dynamic_cast <LIFGap*> (postHyPerLayer);
   if (postLIFGap == NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: postsynaptic layer must be a LIFGap or LIFGap-derived layer.\n",
               parent->parameters()->groupKeywordFromName(name), name);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
   int status = HyPerConn::allocateDataStructures();
   //TODO!!! terrible hack here: should compute sum of gap junctions connection strengths into each post synaptic cell
   // instead, we check that normalize is true as a stop gap
   assert(this->normalizer->getNormalizeFromPostPerspectiveFlag());
   float gap_strength = normalizer->getStrength(); // normalizer->getStrength() / this->postSynapticLayer()->getNumNeurons() * this->preSynapticLayer()->getNumNeurons();
   postLIFGap->addGapStrength(gap_strength);

   return status;
}

} /* namespace PV */
