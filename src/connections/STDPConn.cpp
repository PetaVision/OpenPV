/*
 * STDPConn.cpp
 *
 *  Created on: Jan 28, 2011
 *      Author: sorenrasmussen
 */

#include "STDPConn.hpp"
#include "../layers/LIF.hpp"
#include <assert.h>

namespace PV {

STDPConn::STDPConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
                   ChannelType channel) : HyPerConn(name, hc, pre, post, channel)
{
   initialize();
}

STDPConn::~STDPConn()
{
   // TODO Auto-generated destructor stub
}

/*
 * Using a dynamic_cast operator to convert (downcast) a pointer to a base class (HyPerLayer)
 * to a pointer to a derived class (LIF). This way I do not need to define a virtual
 * function getWmax() in HyPerLayer which only returns a NULL pointer in the base class.
 */
int STDPConn::initialize()
{
   // STDP parameters for modifying weights
   this->pIncr = NULL;
   this->pDecr = NULL;
   this->ampLTP = 1.0;
   this->ampLTD = 1.1;
   this->tauLTP = 20;
   this->tauLTD = 20;
   this->dWMax = 0.1;
   this->localWmaxFlag = false;

   int arbor = 0;
   int numPatches = numWeightPatches(arbor);

   pIncr = createWeights(NULL, numPatches, nxp, nyp, nfp);
   assert(pIncr != NULL);
   pDecr = pvcube_new(&post->getCLayer()->loc, post->getNumExtended());
   assert(pDecr != NULL);

   if (localWmaxFlag){
      LIF * LIF_layer = dynamic_cast<LIF *>(post);
      assert(LIF_layer != NULL);
      Wmax = LIF_layer->getWmax();
      assert(Wmax != NULL);
   } else {
      Wmax = NULL;
   }

   return 0;
}

int STDPConn::initializeThreadBuffers()
{
   return 0;
}

int STDPConn::initializeThreadKernels()
{
   return 0;
}

PVPatch * STDPConn::getPlasticityIncrement(int k, int arbor)
{
   // a separate arbor/patch of plasticity for every neuron
   return pIncr[k];
}

} // End of namespace PV

