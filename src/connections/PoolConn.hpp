/*
 * PoolConn.hpp
 *
 *  Created on: Apr 7, 2009
 *      Author: rasmussn
 */

#ifndef POOLCONN_HPP_
#define POOLCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class PoolConn: public PV::HyPerConn {
public:
   PoolConn(const char * name,
            HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel);

   PVPatch* weights(int k)
   {
      return wPatches[k];
   }

private:
   virtual int initializeWeights(const char * filename);
   int poolWeights(PVPatch * wp, int fPre, int xScale, int yScale, float strength);
};

}

#endif /* POOLCONN_HPP_ */
