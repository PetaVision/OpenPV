/*
 * initWindowed3DGaussWeights.hpp
 *
 *  Created on: Jan 18, 2012
 *      Author: kpeterson
 */

#ifndef INITWINDOWED3DGAUSSWEIGHTS_HPP_
#define INITWINDOWED3DGAUSSWEIGHTS_HPP_

#include "Init3DGaussWeights.hpp"
#include "InitWeights.hpp"
#include "InitWeightsParams.hpp"
#include "InitWindowed3DGaussWeightsParams.hpp"

namespace PV {

class InitWeightsParams;
class InitWindowed3DGaussWeightsParams;

class InitWindowed3DGaussWeights: public PV::Init3DGaussWeights {
public:
   InitWindowed3DGaussWeights();
   virtual ~InitWindowed3DGaussWeights();

   virtual int calcWeights(PVPatch * patch, int patchIndex, int arborId,
         InitWeightsParams *weightParams);
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);

private:
   int windowWeights(PVPatch * patch, InitWindowed3DGaussWeightsParams * weightParamPtr);


};

} /* namespace PV */
#endif /* INITWINDOWED3DGAUSSWEIGHTS_HPP_ */
