/*
 * STDPConn.hpp
 */

#ifndef STDPCONN_HPP_
#define STDPCONN_HPP_

#include "src/connections/HyPerConn.hpp"

namespace PV {

class STDPConn: public PV::HyPerConn {
public:
   STDPConn(const char * name,
            HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel);

   virtual int outputState(FILE * fp, int k);

   PVPatch* weights(int k)
   {
      return wPatches[k];
   }

private:
   virtual int initializeWeights(const char * filename);
   int ruleWeights(PVPatch * wp, int fPre, int xScale, int yScale, float strength);
};

}

#endif /* STDPCONN_HPP_ */
