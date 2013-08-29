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
#include "../columns/GaussianRandom.hpp"

namespace PV {

class InitGaussianRandomWeights: public PV::InitRandomWeights {
public:
   InitGaussianRandomWeights();
   virtual ~InitGaussianRandomWeights();

   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);

protected:
   virtual int initRNGs(HyPerConn * conn, bool isKernel);
   virtual int randomWeights(pvdata_t * patchDataStart, InitWeightsParams *weightParamPtr, int patchIndex);

private:
   int initialize_base();

// Member variables
protected:
   GaussianRandom * gaussianRandState; // Use this instead of randState to use Box-Muller transformation.
};

} /* namespace PV */
#endif /* INITGAUSSIANRANDOMWEIGHTS_HPP_ */
