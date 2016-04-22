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
   InitIdentWeights(char const * name, HyPerCol * hc);
   virtual ~InitIdentWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId);
   virtual InitWeightsParams * createNewWeightParams();
   void calcOtherParams(int patchIndex);


protected:
   InitIdentWeights();
   int initialize(char const * name, HyPerCol * hc);

protected:
   int initialize_base();
}; // class InitIdentWeights

BaseObject * createInitIdentWeights(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* INITIDENTWEIGHTS_HPP_ */
