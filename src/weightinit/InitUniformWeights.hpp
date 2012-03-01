/*
 * InitUniformWeights.hpp
 *
 *  Created on: Aug 23, 2011
 *      Author: kpeterson
 */

#ifndef INITUNIFORMWEIGHTS_HPP_
#define INITUNIFORMWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitUniformWeightsParams.hpp"

namespace PV {

class InitUniformWeights: public PV::InitWeights {
public:
   InitUniformWeights();
   virtual ~InitUniformWeights();
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId, InitWeightsParams *weightParams);

protected:
   virtual int initialize_base();

private:
   int uniformWeights(/* PVPatch * patch */ pvdata_t * dataStart, float iWeight, InitUniformWeightsParams *weightParamPtr);

};

} /* namespace PV */
#endif /* INITUNIFORMWEIGHTS_HPP_ */
