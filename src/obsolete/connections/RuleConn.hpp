/*
 * RuleConn.hpp
 *
 *  Created on: Apr 5, 2009
 *      Author: rasmussn
 */
#ifdef OBSOLETE // Use KernelConn or HyperConn and set the param "weightInitType" to "RuleWeight" in the params file

#ifndef RULECONN_HPP_
#define RULECONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class RuleConn: public PV::HyPerConn {
public:
   RuleConn(const char * name,
            HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel);

   PVPatch * weights(int k)
   {
      int arbor = 0;
      return wPatches[arbor][k];
   }

   virtual PVPatch ** initializeWeights(PVPatch ** patches, int numPatches, const char * filename);
   virtual int ruleWeights(PVPatch * wp, int kPre, int xScale, int yScale, float strength);

protected:
   RuleConn();

private:

};

}

#endif /* RULECONN_HPP_ */
#endif
