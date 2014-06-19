/*
 * InitGauss2DWeights.hpp
 *
 *  Created on: Apr 8, 2013
 *      Author: garkenyon
 */

#ifndef INITGAUSS2DWEIGHTS_HPP_
#define INITGAUSS2DWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitGauss2DWeightsParams.hpp"

namespace PV {

class InitGauss2DWeights: public PV::InitWeights {
public:
   InitGauss2DWeights(HyPerConn * conn);
   virtual ~InitGauss2DWeights();

   virtual InitWeightsParams * createNewWeightParams();

   virtual int calcWeights(pvwdata_t * dataStart, int patchIndex, int arborId);

protected:
   InitGauss2DWeights();
   int initialize_base();
   int initialize(HyPerConn * conn);

private:
   int gauss2DCalcWeights(pvwdata_t * dataStart, InitGauss2DWeightsParams *weightParamPtr);

}; // class InitGauss2DWeights

} /* namespace PV */
#endif /* INITGAUSS2DWEIGHTS_HPP_ */
