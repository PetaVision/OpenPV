/*
 * STDPConn.hpp
 *
 *  Created on: Jan 28, 2011
 *      Author: sorenrasmussen
 */

#ifndef STDPCONN_HPP_
#define STDPCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class STDPConn : HyPerConn {
public:
   STDPConn();
   STDPConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             ChannelType channel);
   virtual ~STDPConn();

   virtual int initializeThreadBuffers();
   virtual int initializeThreadKernels();



};

}

#endif /* STDPCONN_HPP_ */
