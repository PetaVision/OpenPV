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
   InitGaborWeights(HyPerConn * conn);
   virtual ~InitGaborWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId);
   virtual InitWeightsParams * createNewWeightParams();
   void calcOtherParams(PVPatch * patch, int patchIndex);


protected:
   InitGaborWeights();
   int initialize(HyPerConn * conn);

private:
   int initialize_base();
   int gaborWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitGaborWeightsParams * weightParamPtr);
};

} /* namespace PV */
#endif /* INITGABORWEIGHTS_HPP_ */
