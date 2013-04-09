/*
 * InitPoolWeights.hpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#ifndef INITPOOLWEIGHTS_HPP_
#define INITPOOLWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitGauss2DWeights.hpp"

namespace PV {

class InitWeightsParams;
class InitPoolWeightsParams;

class InitPoolWeights: public PV::InitGauss2DWeights {
public:
   InitPoolWeights();
   virtual ~InitPoolWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId,
         InitWeightsParams *weightParams);
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);
   // void calcOtherParams(PVPatch * patch, int patchIndex);


protected:
   int initialize_base();

private:
   int poolWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitPoolWeightsParams * weightParamPtr);
};

} /* namespace PV */
#endif /* INITPOOLWEIGHTS_HPP_ */
