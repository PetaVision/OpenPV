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
   InitGauss2DWeights(char const * name, HyPerCol * hc);
   virtual ~InitGauss2DWeights();

   virtual InitWeightsParams * createNewWeightParams();

   virtual int calcWeights(pvwdata_t * dataStart, int patchIndex, int arborId);

protected:
   InitGauss2DWeights();
   int initialize_base();
   int initialize(char const * name, HyPerCol * hc);

private:
   int gauss2DCalcWeights(pvwdata_t * dataStart, InitGauss2DWeightsParams *weightParamPtr);

}; // class InitGauss2DWeights

BaseObject * createInitGauss2DWeights(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* INITGAUSS2DWEIGHTS_HPP_ */
