/*
 * InitGaborWeights.hpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#ifndef INITGABORWEIGHTS_HPP_
#define INITGABORWEIGHTS_HPP_

#include "InitWeights.hpp"

namespace PV {

class InitWeightsParams;
class InitGaborWeightsParams;

class InitGaborWeights: public PV::InitWeights {
public:
   InitGaborWeights();
   virtual ~InitGaborWeights();

   virtual int calcWeights(PVPatch * patch, int patchIndex,
         InitWeightsParams *weightParams);
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);
   void calcOtherParams(PVPatch * patch, int patchIndex);


protected:
   virtual int initialize_base();

private:
   int gaborWeights(PVPatch * patch, InitGaborWeightsParams * weightParamPtr);
};

} /* namespace PV */
#endif /* INITGABORWEIGHTS_HPP_ */
