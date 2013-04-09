/*
 * InitGaborWeights.hpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#ifndef INITGABORWEIGHTS_HPP_
#define INITGABORWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitGauss2DWeights.hpp"

namespace PV {

class InitWeightsParams;
class InitGaborWeightsParams;

class InitGaborWeights: public PV::InitGauss2DWeights {
public:
   InitGaborWeights();
   virtual ~InitGaborWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId,
         InitWeightsParams *weightParams);
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);
   void calcOtherParams(PVPatch * patch, int patchIndex);


protected:
   int initialize_base();

private:
   int gaborWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitGaborWeightsParams * weightParamPtr);
};

} /* namespace PV */
#endif /* INITGABORWEIGHTS_HPP_ */
