/*
 * SLPError.hpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist
 */

#ifndef SLPERROR_HPP_
#define SLPERROR_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class SLPError: public PV::ANNLayer {
public:
   SLPError(const char * name, HyPerCol * hc);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_classError(enum ParamsIOFlag ioFlag);
protected:
   virtual int updateState(double time, double dt);
private:
   int initialize_base();
   bool classError;
};

} /* namespace PV */
#endif /* ANNERRORLAYER_HPP_ */
