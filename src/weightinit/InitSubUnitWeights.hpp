/*
 * InitSubUnitWeights.hpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#ifndef INITSUBUNITWEIGHTS_HPP_
#define INITSUBUNITWEIGHTS_HPP_

#include "InitWeights.hpp"

namespace PV {

class InitWeightsParams;
class InitSubUnitWeightsParams;


class InitSubUnitWeights: public PV::InitWeights {
public:
   InitSubUnitWeights();
   virtual ~InitSubUnitWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId,
         InitWeightsParams *weightParams);
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);
   // void calcOtherParams(PVPatch * patch, int patchIndex);


protected:
   virtual int initialize_base();

private:
   int subUnitWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitSubUnitWeightsParams * weightParamPtr);
};

} /* namespace PV */
#endif /* INITSUBUNITWEIGHTS_HPP_ */
