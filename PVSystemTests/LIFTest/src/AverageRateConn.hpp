/*
 * AverageRateConn.hpp
 *
 *  Created on: Aug 24, 2012
 *      Author: pschultz
 */

#ifndef AVERAGERATECONN_HPP_
#define AVERAGERATECONN_HPP_

#include <connections/IdentConn.hpp>

namespace PV {

class AverageRateConn : public IdentConn {
public:
   AverageRateConn(const char * name, HyPerCol * hc);
   virtual ~AverageRateConn();
   virtual int updateState(double timed, double dt);

protected:
   AverageRateConn();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();
};

BaseObject * createAverageRateConn(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* AVERAGERATECONN_HPP_ */
