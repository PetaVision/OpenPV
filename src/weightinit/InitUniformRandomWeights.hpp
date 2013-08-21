/*
 * InitUniformRandomWeights.hpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#ifndef INITUNIFORMRANDOMWEIGHTS_HPP_
#define INITUNIFORMRANDOMWEIGHTS_HPP_

#include "InitRandomWeights.hpp"
#include "InitUniformRandomWeightsParams.hpp"
#include "../utils/cl_random.h"

namespace PV {

class InitUniformRandomWeights: public PV::InitRandomWeights {
public:
   InitUniformRandomWeights();
   virtual ~InitUniformRandomWeights();

   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);

protected:
   int randomWeights(pvdata_t * patchDataStart, InitWeightsParams *weightParams, uint4 * rnd_state);
   unsigned int rand_ul(uint4 * state);

private:
   int initialize_base();

};

} /* namespace PV */
#endif /* INITUNIFORMRANDOMWEIGHTS_HPP_ */
