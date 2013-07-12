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
   AverageRateConn(const char * name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name);
   virtual ~AverageRateConn();
   virtual int setParams(PVParams * inputParams);
   virtual int updateState(double timed, double dt);

protected:
   AverageRateConn();
   int initialize(const char * name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name);

private:
   int initialize_base();
};

} /* namespace PV */
#endif /* AVERAGERATECONN_HPP_ */
