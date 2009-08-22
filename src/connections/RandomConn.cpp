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
{
   this->connId = hc->numberOfConnections();
   this->name   = strdup(name);
   this->parent = hc;

   this->numAxonalArborLists = 1;

   initialize(NULL, pre, post, channel);

   hc->addConnection(this);
}

int RandomConn::initializeWeights(const char * filename)
{
   assert(filename == NULL);
   return initializeRandomWeights(0);
}

}
