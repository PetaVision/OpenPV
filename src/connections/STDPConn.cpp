/*
 * STDPConn.cpp
 *
 *  Created on: Jan 28, 2011
 *      Author: sorenrasmussen
 */

#include "STDPConn.hpp"

namespace PV {

STDPConn::STDPConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
                   ChannelType channel) : HyPerConn(name, hc, pre, post, channel)
{

}

STDPConn::~STDPConn()
{
   // TODO Auto-generated destructor stub
}

int STDPConn::initializeThreadBuffers()
{
   return 0;
}

int STDPConn::initializeThreadKernels()
{
   return 0;
}

} // End of namespace PV

