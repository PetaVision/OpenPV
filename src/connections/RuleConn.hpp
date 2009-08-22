/*
 * RuleConn.hpp
 *
 *  Created on: Apr 5, 2009
 *      Author: rasmussn
 */

#ifndef RULECONN_HPP_
#define RULECONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class RuleConn: public PV::HyPerConn {
public:
   RuleConn(const char * name,
            HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel);

   PVPatch * weights(int k)
   {
      int arbor = 0;
      return wPatches[arbor][k];
   }

private:
   virtual int initializeWeights(const char * filename);
   int ruleWeights(PVPatch * wp, int fPre, int xScale, int yScale, float strength);
};

}

#endif /* RULECONN_HPP_ */
