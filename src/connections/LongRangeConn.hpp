/*
 * LongRangeConn.hpp
 *
 *  Created on: Feb 23, 2009
 *      Author: rasmussn
 */

#ifndef LONGRANGECONN_HPP_
#define LONGRANGECONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class LongRangeConn: public PV::HyPerConn {
public:
   LongRangeConn(const char * name,
                 HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel);

protected:
   virtual int initializeWeights(const char * filename);

   virtual int createAxonalArbors();

   int calcWeights(PVPatch * wp, int kPre, int no, int xScale, int yScale,
                   float aspect, float sigma, float r2Max, float lambda, float strength);

};

}

#endif /* LONGRANGECONN_HPP_ */
