/*
 * VaryingHyPerConn.hpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#ifndef VARYINGHYPERCONN_HPP_
#define VARYINGHYPERCONN_HPP_

#include "../PetaVision/src/connections/HyPerConn.hpp"

namespace PV {

class VaryingHyPerConn : public HyPerConn {

public:
   VaryingHyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
         ChannelType channel, const char * filename, InitWeights *weightInit);
   virtual ~VaryingHyPerConn();
   virtual int setParams(PVParams * inputParams);
   virtual int updateWeights(int axonId = 0);

protected:
   int initialize(const char * name, HyPerCol * hc,
                  HyPerLayer * pre, HyPerLayer * post,
                  ChannelType channel, const char * filename,
                  InitWeights *weightInit=NULL);
   virtual int calc_dW(int axonId);

}; // end class VaryingHyPerConn

}  // end namespace PV block


#endif /* VARYINGHYPERCONN_HPP_ */
