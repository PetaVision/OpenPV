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

namespace PV {

class InitUniformRandomWeights: public PV::InitWeights {
public:
   InitUniformRandomWeights();
   virtual ~InitUniformRandomWeights();

   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);

   virtual int calcWeights(/* PVPatch * wp */ pvdata_t * dataStart, int patchIndex, int arborId, InitWeightsParams *weightParams);

protected:
   int initialize_base();

private:
   int uniformWeights(/* PVPatch * wp */ pvdata_t * dataStart, float minwgt, float maxwgt, float sparseFraction, InitUniformRandomWeightsParams *weightParamPtr);
};

} /* namespace PV */
#endif /* INITUNIFORMRANDOMWEIGHTS_HPP_ */
