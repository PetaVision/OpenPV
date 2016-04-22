/*
 * InitOneToOneWeights.hpp
 *
 *  Created on: Sep 28, 2011
 *      Author: kpeterson
 */

#ifndef INITONETOONEWEIGHTS_HPP_
#define INITONETOONEWEIGHTS_HPP_

#include "InitWeights.hpp"

namespace PV {

class InitWeightsParams;
class InitOneToOneWeightsParams;

// TODO make InitOneToOneWeights a derived class of InitUniformWeights
class InitOneToOneWeights: public PV::InitWeights {
public:
   InitOneToOneWeights(char const * name, HyPerCol * hc);
   virtual ~InitOneToOneWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId);
   virtual InitWeightsParams * createNewWeightParams();
   void calcOtherParams(int patchIndex);


protected:
   InitOneToOneWeights();
   int initialize(char const * name, HyPerCol * hc);
   int createOneToOneConnection(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, float iWeight, InitWeightsParams * weightParamPtr);

private:
   int initialize_base();
}; // class InitOneToOneWeights

BaseObject * createInitOneToOneWeights(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* INITONETOONEWEIGHTS_HPP_ */
