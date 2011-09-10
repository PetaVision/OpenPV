/*
 * CliqueConn.cpp
 *
 *  Created on: Sep 8, 2011
 *      Author: gkenyon
 */

#include "CliqueConn.hpp"

namespace PV {

CliqueConn::CliqueConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
            ChannelType channel, const char * filename) : KernelConn(name, hc, pre, post,
                  channel, filename){};
CliqueConn::CliqueConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
            ChannelType channel, const char * filename, InitWeights *weightInit) : KernelConn(name, hc, pre, post,
                  channel, filename, weightInit){};
CliqueConn::CliqueConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
            ChannelType channel) : KernelConn(name, hc, pre, post,
                  channel){};

int CliqueConn::updateState(float time, float dt){
   int status = KernelConn::updateState(time,dt);
   assert(status == PV_SUCCESS);
   return PV_SUCCESS;
};

int CliqueConn::updateWeights(int arborId){

   return PV_CONTINUE;};

PVPatch ** CliqueConn::normalizeWeights(PVPatch ** patches, int numPatches, int arborId){
   return PV_SUCCESS;};

} /* namespace PV */
