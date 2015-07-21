/*
 * MaxPooling.hpp
 *
 *  Created on: Jan 3, 2013
 *      Author: gkenyon
 */

#ifndef MAXPOOLING_HPP_
#define MAXPOOLING_HPP_

#include "ANNLayer.hpp"

namespace PV {

class MaxPooling: public PV::HyPerLayer {
public:
   MaxPooling(const char * name, HyPerCol * hc);
   virtual ~MaxPooling();
   virtual int recvSynapticInput(HyPerConn * conn, const PVLayerCube * activity, int axonId);
protected:
   MaxPooling();
   int initialize(const char * name, HyPerCol * hc);
private:
   int initialize_base();
};

} /* namespace PV */
#endif /* MAXPOOLING_HPP_ */
