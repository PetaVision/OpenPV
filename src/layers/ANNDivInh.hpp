/*
 * ANNDivInh.hpp
 *
 *  Created on: Jan 22, 2012
 *      Author: kpeterson
 */

#ifndef ANNDIVINH_HPP_
#define ANNDIVINH_HPP_

#include "ANNLayer.hpp"

namespace PV {

class ANNDivInh: public PV::ANNLayer {
public:
   ANNDivInh(const char* name, HyPerCol * hc, int numChannels);
   ANNDivInh(const char* name, HyPerCol * hc);
   virtual ~ANNDivInh();

   virtual int updateV();

protected:
   ANNDivInh();
   int initialize(const char * name, HyPerCol * hc, int numChannels=MAX_CHANNELS);

private:
   int initialize_base();

};

} /* namespace PV */
#endif /* ANNDIVINH_HPP_ */
