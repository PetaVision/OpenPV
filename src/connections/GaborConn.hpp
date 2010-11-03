/*
 * GaborConn.h
 *
 *  Created on: Jan 12, 2009
 *      Author: rasmussn
 */

#ifndef GABORCONN_H_
#define GABORCONN_H_

#include "KernelConn.hpp"

namespace PV {

class GaborConn: public PV::KernelConn {
public:
   GaborConn(const char * name,
             HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel);

   int gaborWeights(PVPatch * wp, int xScale, int yScale,
                    float aspect, float sigma, float r2Max, float lambda, float strength);

protected:
   PVPatch ** initializeDefaultWeights(PVPatch ** patches, int numPatches);
   PVPatch ** initializeGaborWeights(PVPatch ** patches, int numPatches);
};

}

#endif /* GABORCONN_H_ */
