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
	InitGauss2DWeights();
	virtual ~InitGauss2DWeights();

	virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);

	virtual int calcWeights(/* PVPatch * wp */pvdata_t * dataStart,
			int patchIndex, int arborId, InitWeightsParams *weightParams);

protected:
	int initialize_base();

private:
	int gauss2DCalcWeights(pvdata_t * dataStart,
			InitGauss2DWeightsParams *weightParamPtr);

}; // calss InitGauss2DWeights

} /* namespace PV */
#endif /* INITGAUSS2DWEIGHTS_HPP_ */
