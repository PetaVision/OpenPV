/*
 * CocircConn.h
 *
 *  Created on: Nov 10, 2008
 *      Author: rasmussn
 */

#ifndef COCIRCCONN_HPP_
#define COCIRCCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class CocircConn: public PV::HyPerConn {
public:
   CocircConn(const char * name,
              HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel);

   PVPatch* weights(int k)
   {
      return wPatches[k];
   }

private:
   virtual int initializeWeights(const char * filename);
   int cocircWeights(PVPatch * wp, int fPre, int xScale, int yScale,
                     float sigma, float r2Max, float strength);

};

}

#endif /* COCIRCCONN_HPP_ */
