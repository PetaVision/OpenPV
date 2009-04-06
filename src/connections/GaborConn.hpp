/*
 * GaborConn.h
 *
 *  Created on: Jan 12, 2009
 *      Author: rasmussn
 */

#ifndef GABORCONN_H_
#define GABORCONN_H_

#include "HyPerConn.hpp"

namespace PV {

class GaborConn: public PV::HyPerConn {
public:
   GaborConn(const char * name,
             HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel);

   int gaborWeights(PVPatch * wp, int xScale, int yScale,
                    float aspect, float sigma, float r2Max, float lambda, float strength);

protected:
   virtual int initializeWeights(const char * filename);

};

}

#endif /* GABORCONN_H_ */
