/*
 * GapConn.hpp
 *
 *  Created on: Aug 2, 2011
 *      Author: garkenyon
 */

#ifndef GAPCONN_HPP_
#define GAPCONN_HPP_

#include "KernelConn.hpp"

namespace PV {

class GapConn: public PV::KernelConn {
public:
   GapConn();
   GapConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
         HyPerLayer * post, ChannelType channel, const char * filename, InitWeights *weightInit=NULL);
   virtual ~GapConn();
   virtual int initNormalize();
private:
   bool initNormalizeFlag;

};

} /* namespace PV */
#endif /* GAPCONN_HPP_ */
