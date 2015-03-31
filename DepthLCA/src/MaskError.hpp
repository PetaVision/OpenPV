/*
 * MaskError.hpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist
 */

#ifndef MASKERROR_HPP_
#define MASKERROR_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class MaskError: public PV::ANNLayer {
public:
   MaskError(const char * name, HyPerCol * hc);
protected:
   virtual int updateState(double time, double dt);
};

} /* namespace PV */
#endif /* ANNERRORLAYER_HPP_ */
