/*
 * GenerativeConn.cpp
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#include "GenerativeConn.hpp"

namespace PV {

GenerativeConn::GenerativeConn() {

}  // end of GenerativeConn::GenerativeConn()

GenerativeConn::GenerativeConn(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post) {
	   initialize_base();
	   initialize(name, hc, pre, post, channel, NULL);
}  // end of GenerativeConn::GenerativeConn(const char *, HyPerCol *,
   //   HyPerLayer *, HyPerLayer *)

GenerativeConn::GenerativeConn(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, int channel) {
	   initialize_base();
	   initialize(name, hc, pre, post, channel);
}  // end of GenerativeConn::GenerativeConn(const char *, HyPerCol *,
   //   HyPerLayer *, HyPerLayer *, int)

GenerativeConn::GenerativeConn(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, int channel,
        const char * filename) {
	   initialize_base();
	   initialize(name, hc, pre, post, channel, filename);
}  // end of GenerativeConn::GenerativeConn(const char *, HyPerCol *,
   //   HyPerLayer *, HyPerLayer *, int, const char *)

int GenerativeConn::initialize(const char * name, HyPerCol * hc,
		HyPerLayer * pre, HyPerLayer * post, int channel) {
    return initialize(name, hc, pre, post, channel, NULL);
}

int GenerativeConn::initialize(const char * name, HyPerCol * hc,
		HyPerLayer * pre, HyPerLayer * post, int channel,
		const char * filename) {
	HyPerConn::initialize(name, hc, pre, post, channel, filename);
	weightUpdatePeriod = parent->parameters()->value(name, "weightUpdatePeriod", 1.0f);
	nextWeightUpdate = weightUpdatePeriod;
	relaxation = parent->parameters()->value(name, "relaxation", 1.0f);
	return EXIT_SUCCESS;
}

int GenerativeConn::updateState(float time, float dt) {
    if(time > nextWeightUpdate) {
    	nextWeightUpdate += weightUpdatePeriod;
        updateWeights(0);
    }
    return EXIT_SUCCESS;
}  // end of GenerativeConn::updateState(float, float)

int GenerativeConn::updateWeights(int axonID) {
	printf("updateWeights %s\n", name);
    return EXIT_SUCCESS;
}  // end of GenerativeConn::updateWeights(int);

}  // end of namespace PV block
