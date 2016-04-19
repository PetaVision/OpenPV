/*
 * InitSpreadOverArborsWeights.hpp
 *
 *  Created on: Sep 1, 2011
 *      Author: kpeterson
 */

#ifndef INITSPREADOVERARBORSWEIGHTS_HPP_
#define INITSPREADOVERARBORSWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitGauss2DWeights.hpp"

namespace PV {

class InitSpreadOverArborsWeightsParams;


class InitSpreadOverArborsWeights: public PV::InitGauss2DWeights {
public:
   InitSpreadOverArborsWeights(char const * name, HyPerCol * hc);
   virtual ~InitSpreadOverArborsWeights();
   virtual InitWeightsParams * createNewWeightParams();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId);

protected:
   InitSpreadOverArborsWeights();
   int initialize(char const * name, HyPerCol * hc);

private:
   int initialize_base();
   int spreadOverArborsWeights(/* PVPatch * patch */ pvdata_t * dataStart, int arborId,
         InitSpreadOverArborsWeightsParams * weightParamPtr);
};

BaseObject * createInitSpreadOverArborsWeights(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* INITSPREADOVERARBORSWEIGHTS_HPP_ */
