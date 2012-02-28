/*
 * InitSpreadOverArborsWeights.hpp
 *
 *  Created on: Sep 1, 2011
 *      Author: kpeterson
 */

#ifndef INITSPREADOVERARBORSWEIGHTS_HPP_
#define INITSPREADOVERARBORSWEIGHTS_HPP_

#include "InitWeights.hpp"

namespace PV {

class InitSpreadOverArborsWeightsParams;


class InitSpreadOverArborsWeights: public PV::InitWeights {
public:
   InitSpreadOverArborsWeights();
   virtual ~InitSpreadOverArborsWeights();
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId, InitWeightsParams *weightParams);

protected:
   virtual int initialize_base();

private:
   int spreadOverArborsWeights(/* PVPatch * patch */ pvdata_t * dataStart, int arborId,
         InitSpreadOverArborsWeightsParams * weightParamPtr);
};

} /* namespace PV */
#endif /* INITSPREADOVERARBORSWEIGHTS_HPP_ */
