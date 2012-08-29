/*
 * AverageRateConn.hpp
 *
 *  Created on: Aug 24, 2012
 *      Author: pschultz
 */

#ifndef AVERAGERATECONN_HPP_
#define AVERAGERATECONN_HPP_

#include "../PetaVision/src/connections/IdentConn.hpp"

namespace PV {

class AverageRateConn : public IdentConn {
public:
   AverageRateConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post);
   virtual ~AverageRateConn();
   virtual int setParams(PVParams * inputParams);
   virtual int updateState(float timef, float dt);

protected:
   AverageRateConn();
   int initialize(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post);

private:
   int initialize_base();
};

} /* namespace PV */
#endif /* AVERAGERATECONN_HPP_ */
