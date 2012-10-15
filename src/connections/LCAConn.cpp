/*
 * LCAConn.cpp
 *
 *  Created on: Oct 8, 2012
 *      Author: kpatel
 */

#include "LCAConn.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace PV {
  
  LCAConn::LCAConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
		   const char * filename, InitWeights *weightInit, Movie * auxLayer) 
  {
    KernelConn::initialize_base();
    KernelConn::initialize(name, hc, pre, post, filename, weightInit);
    layerOfInterest = auxLayer;
  }

  pvdata_t LCAConn::updateRule_dW(pvdata_t pre, pvdata_t post)
  {
    pvdata_t input = *(layerOfInterest->getImageBuffer());
    return pre*(post-input);
  }
}


