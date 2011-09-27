/*
 * ANNSquaredLayer.hpp
 *
 *  Created on: Sep 21, 2011
 *      Author: kpeterson
 */

#ifndef ANNSQUAREDLAYER_HPP_
#define ANNSQUAREDLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class ANNSquaredLayer: public PV::ANNLayer {
public:
   ANNSquaredLayer(const char* name, HyPerCol * hc, int numChannels);
   ANNSquaredLayer(const char* name, HyPerCol * hc);
   virtual ~ANNSquaredLayer();
   virtual int updateV();

   virtual int squareV();

protected:
   int initialize();

};

} /* namespace PV */
#endif /* ANNSQUAREDLAYER_HPP_ */
