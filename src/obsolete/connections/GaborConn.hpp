/*
 * GaborConn.h
 *
 *  Created on: Jan 12, 2009
 *      Author: rasmussn
 */
#ifdef OBSOLETE // Use KernelConn or GaborConn and set the param "weightInitType" to "GaborWeight" in the params file

#ifndef GABORCONN_H_
#define GABORCONN_H_

#include "KernelConn.hpp"

namespace PV {

class GaborConn: public PV::KernelConn {
public:
   GaborConn(const char * name,
             HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel);

   int gaborWeights(PVPatch * wp, int xScale, int yScale,
                    float aspect, float sigma, float r2Max, float lambda, float strength, float phi);

protected:
   PVPatch ** initializeDefaultWeights(PVPatch ** patches, int numPatches);
   PVPatch ** initializeGaborWeights(PVPatch ** patches, int numPatches);
};

}

#endif /* GABORCONN_H_ */
#endif
