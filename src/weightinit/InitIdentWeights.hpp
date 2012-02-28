/*
 * InitIdentWeights.hpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#ifndef INITIDENTWEIGHTS_HPP_
#define INITIDENTWEIGHTS_HPP_

#include "InitOneToOneWeights.hpp"

namespace PV {

class InitWeightsParams;
class InitIdentWeightsParams;

class InitIdentWeights: public PV::InitOneToOneWeights {
public:
   InitIdentWeights();
   virtual ~InitIdentWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId,
         InitWeightsParams *weightParams);
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);
   void calcOtherParams(int patchIndex);


protected:
   virtual int initialize_base();
};

} /* namespace PV */
#endif /* INITIDENTWEIGHTS_HPP_ */
