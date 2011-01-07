/*
 * STDPConn.hpp
 */

#ifndef BICONN_HPP_
#define BICONN_HPP_

#include "src/connections/RuleConn.hpp"

namespace PV {

class BiConn: public PV::RuleConn {
public:
   BiConn(const char * name,
          HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
          int type);

   virtual int outputState(FILE * fp, int k);

   PVPatch* weights(int k)
   {
      int arbor = 0;
      return wPatches[arbor][k];
   }

   virtual int ruleWeights(PVPatch * wp, int fPre, int xScale, int yScale, float strength);

private:

   int type;
};

}

#endif /* BICONN_HPP_ */
