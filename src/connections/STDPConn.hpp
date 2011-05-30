/*
 * STDPConn.hpp
 *
 *  Created on: Jan 28, 2011
 *      Author: sorenrasmussen
 */

#ifndef STDPCONN_HPP_
#define STDPCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class STDPConn : HyPerConn {
public:
   STDPConn();
   STDPConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             ChannelType channel);
   virtual ~STDPConn();

   virtual int initializeThreadBuffers();
   virtual int initializeThreadKernels();

   virtual PVPatch     * getPlasticityIncrement(int k, int arbor);
   inline  PVLayerCube * getPlasticityDecrement()     {return pDecr;}

protected:

   int initialize();

   PVLayerCube    * pDecr;      // plasticity decrement variable (Mi) for post-synaptic layer
   PVPatch       ** pIncr;      // list of stdp patches Psij variable

   bool     localWmaxFlag;  // presence of rate dependent wMax;
   pvdata_t * Wmax;   // adaptive upper STDP weight boundary

   // STDP parameters for modifying weights
   float ampLTP; // long term potentiation amplitude
   float ampLTD; // long term depression amplitude
   float tauLTP;
   float tauLTD;
   float dWMax;
};

}

#endif /* STDPCONN_HPP_ */
