/*
 * RandomConn.hpp
 *
 *  Created on: Apr 27, 2009
 *      Author: rasmussn
 */

#ifndef RANDOMCONN_HPP_
#define RANDOMCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class RandomConn: public PV::HyPerConn {
public:
   RandomConn(const char * name,
              HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel);

   virtual int initializeWeights(const char * filename);
};

}

#endif /* RANDOMCONN_HPP_ */
