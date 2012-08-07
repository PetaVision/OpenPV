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
   GapConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
         HyPerLayer * post, const char * filename, InitWeights *weightInit=NULL);
   // No channel argument to constructor because GapConn must always use CHANNEL_GAP
   virtual ~GapConn();
protected:
   GapConn();
private:
   virtual int initNormalize();
   bool initNormalizeFlag;

};

} /* namespace PV */
#endif /* GAPCONN_HPP_ */
