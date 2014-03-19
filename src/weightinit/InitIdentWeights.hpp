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
   InitIdentWeights(HyPerConn * conn);
   virtual ~InitIdentWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId);
   virtual InitWeightsParams * createNewWeightParams();
   void calcOtherParams(int patchIndex);


protected:
   InitIdentWeights();
   int initialize(HyPerConn * conn);

protected:
   int initialize_base();
};

} /* namespace PV */
#endif /* INITIDENTWEIGHTS_HPP_ */
