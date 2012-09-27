/*
 * LCAConn.hpp
 *
 *  Created on: Sep 26, 2012
 *      Author: slundquist
 */

#ifndef LCACONN_HPP_
#define LCACONN_HPP_

#include "HyPerConn.hpp"
#include "../include/default_params.h"
#include <stdio.h>

namespace PV {

class LCAConn : HyPerConn {
   LCAConn();
   LCAConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
            const char * filename=NULL, bool stdpFlag=true,
            InitWeights *weightInit=NULL);
   virtual ~LCAConn();

   int setParams(PVParams * params);
};



#endif /* LCACONN_HPP_ */
