/*
 * InitIdentWeights.hpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#ifndef INITIDENTWEIGHTS_HPP_
#define INITIDENTWEIGHTS_HPP_

#include "InitWeights.hpp"

namespace PV {

class InitWeightsParams;
class InitIdentWeightsParams;

class InitIdentWeights: public PV::InitWeights {
public:
   InitIdentWeights();
   virtual ~InitIdentWeights();

   virtual int calcWeights(PVPatch * patch, int patchIndex,
         InitWeightsParams *weightParams);
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);
   void calcOtherParams(PVPatch * patch, int patchIndex);


protected:
   virtual int initialize_base();
};

} /* namespace PV */
#endif /* INITIDENTWEIGHTS_HPP_ */
