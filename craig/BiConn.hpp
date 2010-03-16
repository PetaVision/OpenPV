/*
 * STDPConn.hpp
 */

#ifndef BICONN_HPP_
#define BICONN_HPP_

#include "src/connections/HyPerConn.hpp"

namespace PV {

class BiConn: public PV::HyPerConn {
public:
   BiConn(const char * name,
          HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel,
          int type);

   virtual int outputState(FILE * fp, int k);

   PVPatch* weights(int k)
   {
      int arbor = 0;
      return wPatches[arbor][k];
   }

private:
   PVPatch ** initializeWeights(PVPatch ** patches,
                                int numPatches, const char * filename);
   int ruleWeights(PVPatch * wp, int kPre, float strength);

   int type;
};

}

#endif /* BICONN_HPP_ */
