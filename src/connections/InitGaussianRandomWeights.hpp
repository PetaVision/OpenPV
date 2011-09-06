/*
 * InitGaussianRandomWeights.hpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#ifndef INITGAUSSIANRANDOMWEIGHTS_HPP_
#define INITGAUSSIANRANDOMWEIGHTS_HPP_

#include "InitWeights.hpp"

namespace PV {

class InitGaussianRandomWeights: public PV::InitWeights {
public:
	InitGaussianRandomWeights();
//	InitGaussianRandomWeights(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
//		 ChannelType channel);
	virtual ~InitGaussianRandomWeights();

	   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);

	   virtual int calcWeights(PVPatch * patch, int patchIndex, int arborId, InitWeightsParams *weightParams);


protected:
	virtual int initialize_base();
//	int initialize(const char * name, HyPerCol * hc,
//	               HyPerLayer * pre, HyPerLayer * post,
//	               ChannelType channel);

private:
	int gaussianWeights(PVPatch * wp, float mean, float stdev);

};

} /* namespace PV */
#endif /* INITGAUSSIANRANDOMWEIGHTS_HPP_ */
