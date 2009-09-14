/*
 * RandomConn.cpp
 *
 *  Created on: Apr 27, 2009
 *      Author: rasmussn
 */

#include "RandomConn.hpp"
#include <assert.h>
#include <string.h>

namespace PV {

RandomConn::RandomConn(const char * name,
                       HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel)
          : HyPerConn(name, hc, pre, post, channel, PROTECTED_NUMBER)
{
   this->numAxonalArborLists = 1;
   initialize();
   hc->addConnection(this);
}

int RandomConn::initializeWeights(const char * filename)
{
   assert(filename == NULL);
   return initializeRandomWeights(0);
}

} // namespace PV
