/*
 * LCAConn.hpp
 *
 *  Created on: Oct 8, 2012
 *      Author: kpatel
 */

#ifndef LCACONN_HPP_
#define LCACONN_HPP_

#include "KernelConn.hpp"
#include "../include/pv_common.h"
#include "../layers/Movie.hpp"

namespace PV {

class LCAConn : KernelConn {

public:
  LCAConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
	  const char * filename=NULL, InitWeights *weightInit=NULL, Movie * auxLayer=NULL);
  
protected:
  int initialize(const char * name, HyPerCol * hc,
		 HyPerLayer * pre, HyPerLayer * post,
		 const char * filename,
		 InitWeights *weightInit=NULL,
		 Movie * auxLayer=NULL);
  int update_dW(int axonId);
  pvdata_t updateRule_dW(pvdata_t pre, pvdata_t post); // Inherits from KernelConn; without it the rule below causes an overloaded-virtual warning
  pvdata_t updateRule_dW(pvdata_t pre, pvdata_t post, int offset);
  Movie * layerOfInterest;
};

}
#endif /* LCACONN_HPP_ */
