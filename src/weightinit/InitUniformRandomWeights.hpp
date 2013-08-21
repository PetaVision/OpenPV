/*
 * InitUniformRandomWeights.hpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#ifndef INITUNIFORMRANDOMWEIGHTS_HPP_
#define INITUNIFORMRANDOMWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitUniformRandomWeightsParams.hpp"
#include "../utils/cl_random.h"

namespace PV {

class InitUniformRandomWeights: public PV::InitWeights {
public:
   InitUniformRandomWeights();
   virtual ~InitUniformRandomWeights();

   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);

   virtual int calcWeights(pvdata_t * dataStart, int patchIndex, int arborId, InitWeightsParams *weightParams);

protected:
   int initialize_base();
   virtual int initRNGs(HyPerConn * conn, bool isKernel);
   unsigned int rand_ul(uint4 * state);

private:
   int uniformWeights(pvdata_t * dataStart, float minwgt, float maxwgt,
         float sparseFraction, InitUniformRandomWeightsParams *weightParamPtr, int patchIndex);
};

} /* namespace PV */
#endif /* INITUNIFORMRANDOMWEIGHTS_HPP_ */
