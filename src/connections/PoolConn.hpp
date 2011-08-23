/*
 * PoolConn.hpp
 *
 *  Created on: Apr 7, 2009
 *      Author: rasmussn
 */
#ifdef OBSOLETE // Use KernelConn or HyperConn and set the param "weightInitType" to "PoolWeight" in the params file

#ifndef POOLCONN_HPP_
#define POOLCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class PoolConn: public PV::HyPerConn {
public:
   PoolConn(const char * name,
            HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel);

   PVPatch * weights(int k)
   {
      int arbor = 0;
      return wPatches[arbor][k];
   }

private:
   virtual int initializeWeights(const char * filename);
   int poolWeights(PVPatch * wp, int fPre, int xScale, int yScale, float strength);
};

}

#endif /* POOLCONN_HPP_ */
#endif
