/*
 * InitGaussianRandomWeights.hpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#ifndef INITGAUSSIANRANDOMWEIGHTS_HPP_
#define INITGAUSSIANRANDOMWEIGHTS_HPP_

#include "InitRandomWeights.hpp"
#include "InitGaussianRandomWeightsParams.hpp"
#include "../utils/cl_random.h"

namespace PV {

class InitGaussianRandomWeights: public PV::InitRandomWeights {
public:
   InitGaussianRandomWeights();
   virtual ~InitGaussianRandomWeights();

   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);

protected:
   virtual int randomWeights(pvdata_t * patchDataStart, InitWeightsParams *weightParamPtr, uint4 * rnd_state);

private:
   int initialize_base();

};

} /* namespace PV */
#endif /* INITGAUSSIANRANDOMWEIGHTS_HPP_ */
