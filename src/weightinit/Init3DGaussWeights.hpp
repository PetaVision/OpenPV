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
#include "InitGauss2DWeights.hpp"
#include "Init3DGaussWeightsParams.hpp"

namespace PV {

//class InitWeightsParams;
//class Init3DGaussWeightsParams;

class Init3DGaussWeights: public PV::InitGauss2DWeights {
public:
   Init3DGaussWeights(HyPerConn * conn);
   virtual ~Init3DGaussWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId);
   virtual InitWeightsParams * createNewWeightParams();


protected:
   Init3DGaussWeights();
   int initialize_base();
   int initialize(HyPerConn * conn);

//private:
   int gauss3DWeights(/*PVPatch * patch */ pvdata_t * dataStart, Init3DGaussWeightsParams * weightParamPtr);

};

} /* namespace PV */
#endif /* INIT3DGAUSSWEIGHTS_HPP_ */
