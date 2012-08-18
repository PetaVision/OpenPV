/*
 * BIDSConn.hpp
 *
 *  Created on: Aug 17, 2012
 *      Author: Brennan Nowers
 */

#ifndef BIDSCONN_HPP_
#define BIDSCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class BIDSConn : public PV::HyPerConn {

public:
   BIDSConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             ChannelType channel, const char * filename, InitWeights *weightInit);

protected:
   virtual int setPatchSize(const char* filename);

};

} // namespace PV

#endif /* BIDSCONN_HPP_ */
