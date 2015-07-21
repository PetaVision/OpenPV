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
   InitWindowed3DGaussWeights(HyPerConn * conn);
   virtual ~InitWindowed3DGaussWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId);
   virtual InitWeightsParams * createNewWeightParams();

protected:
   InitWindowed3DGaussWeights();
   int initialize(HyPerConn * conn);

private:
   int initialize_base();
   int windowWeights(pvdata_t * dataStart, InitWindowed3DGaussWeightsParams * weightParamPtr);


};

} /* namespace PV */
#endif /* INITWINDOWED3DGAUSSWEIGHTS_HPP_ */
