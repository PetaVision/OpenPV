/*
 * Init3DGaussWeights.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef INIT3DGAUSSWEIGHTS_HPP_
#define INIT3DGAUSSWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitWeightsParams.hpp"
#include "Init3DGaussWeightsParams.hpp"
#include "InitGauss2DWeights.hpp"

namespace PV {

class InitWeightsParams;
class Init3DGaussWeightsParams;

class Init3DGaussWeights: public PV::InitGauss2DWeights {
public:
   Init3DGaussWeights();
   virtual ~Init3DGaussWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId,
         InitWeightsParams *weightParams);
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);


protected:
   int initialize_base();

//private:
   int gauss3DWeights(/*PVPatch * patch */ pvdata_t * dataStart, Init3DGaussWeightsParams * weightParamPtr);

};

} /* namespace PV */
#endif /* INIT3DGAUSSWEIGHTS_HPP_ */
