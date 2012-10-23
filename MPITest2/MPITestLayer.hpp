/*
 * MPITestLayer.hpp
 *
 *  Created on: Sep 27, 2011
 *      Author: gkenyon
 */

#ifndef MPITESTLAYER_HPP_
#define MPITESTLAYER_HPP_

#include "../PetaVision/src/layers/ANNLayer.hpp"

namespace PV {

class MPITestLayer: public PV::ANNLayer {
public:
	MPITestLayer(const char* name, HyPerCol * hc, int numChannels);
	MPITestLayer(const char* name, HyPerCol * hc);
	virtual int updateState(double time, double dt);
	virtual int publish(InterColComm * comm, double timed);
	int setVtoGlobalPos();
	int setActivitytoGlobalPos();

private:
    int initialize(const char * name, HyPerCol * hc, int numChannels);

};

} /* namespace PV */
#endif /* MPITESTLAYER_HPP_ */
