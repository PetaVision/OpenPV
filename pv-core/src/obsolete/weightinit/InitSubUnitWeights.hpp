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
   InitSubUnitWeights(HyPerConn * conn);
   virtual ~InitSubUnitWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId);
   virtual InitWeightsParams * createNewWeightParams();
   // void calcOtherParams(PVPatch * patch, int patchIndex);


protected:
   InitSubUnitWeights();
   int initialize(HyPerConn * conn);

private:
   int initialize_base();
   int subUnitWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitSubUnitWeightsParams * weightParamPtr);
};

} /* namespace PV */
#endif /* INITSUBUNITWEIGHTS_HPP_ */
