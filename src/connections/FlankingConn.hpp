/*
 * FileConn.hpp
 *
 *  Created on: Oct 27, 2008
 *      Author: rasmussn
 */

#ifndef FLANKINGCONN_HPP_
#define FLANKINGCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class FlankingConn: public PV::HyPerConn {
public:
   FlankingConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post);

protected:
   virtual int initializeWeights(const char * filename);

};

}

#endif /* FLANKINGCONN_HPP_ */
