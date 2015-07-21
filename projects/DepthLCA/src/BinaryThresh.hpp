/*
 * BinaryThresh.hpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist
 */

#ifndef BINARYTHRESH_HPP_
#define BINARYTHRESH_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class BinaryThresh: public PV::ANNLayer {
public:
   BinaryThresh(const char * name, HyPerCol * hc);
protected:
   virtual int updateState(double time, double dt);
};

} /* namespace PV */
#endif /* ANNERRORLAYER_HPP_ */
