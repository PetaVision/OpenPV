/*
 * PlasticConnTestLayer.hpp
 *
 *  Created on: Oct 24, 2011
 *      Author: pschultz
 */

#ifndef PLASTICCONNTESTLAYER_HPP_
#define PLASTICCONNTESTLAYER_HPP_

#include "../PetaVision/src/layers/ANNLayer.hpp"

namespace PV {

class PlasticConnTestLayer: public PV::ANNLayer {
public:
	PlasticConnTestLayer(const char* name, HyPerCol * hc, int numChannels);
	PlasticConnTestLayer(const char* name, HyPerCol * hc);
	virtual int updateState(float time, float dt);
	virtual int publish(InterColComm * comm, float time);
protected:
	int copyAtoV();
	int setActivitytoGlobalPos();
	int initialize();
};

} /* namespace PV */
#endif /* PLASTICCONNTESTLAYER_HPP_ */
